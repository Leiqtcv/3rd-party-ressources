/*
 * 6D localization for humanoid robots
 *
 * Copyright 2009-2012 Armin Hornung, University of Freiburg
 * http://www.ros.org/wiki/humanoid_localization
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <humanoid_localization/RaycastingModel.h>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <octomap_ros/conversions.h>

namespace humanoid_localization{

RaycastingModel::RaycastingModel(ros::NodeHandle* nh, boost::shared_ptr<MapModel> mapModel, EngineT * rngEngine)
: ObservationModel(nh, mapModel, rngEngine)
{
  // params:
  nh->param("raycasting/z_hit", m_zHit, 0.8);
  nh->param("raycasting/z_short", m_zShort, 0.1);
  nh->param("raycasting/z_max", m_zMax, 0.05);
  nh->param("raycasting/z_rand", m_zRand, 0.05);
  nh->param("raycasting/sigma_hit", m_sigmaHit, 0.02);
  nh->param("raycasting/lambda_short", m_lambdaShort, 0.1);

  if (m_zMax <= 0.0){
    ROS_ERROR("raycasting/z_max needs to be > 0.0");
  }

  if (m_zRand <= 0.0){
    ROS_ERROR("raycasting/z_rand needs to be > 0.0");
  }
   #pragma omp parallel
   #pragma omp critical
    {
      if (omp_get_thread_num() == 0){
        ROS_INFO("Using %d threads in RaycastingModel", omp_get_num_threads());
      }
    }
}

RaycastingModel::~RaycastingModel(){

}

void RaycastingModel::integrateMeasurement(Particles& particles, const PointCloud& pc, const std::vector<float>& ranges, float max_range, const tf::Transform& base_to_laser){
  assert(pc.size() == ranges.size());

  //for RMS calculation: missSqd equals the distance value that is represented by the probability p of a missed ray: p = m_zMax / max_range
  double missSqd = log((m_zMax / max_range)/m_zHit*SQRT_2_PI*m_sigmaHit)*(-2*m_sigmaHit*m_sigmaHit);
//  std::cout << "Missing Distance Squared = " << missSqd << std::endl;

  if (!m_map){
    ROS_ERROR("Map file is not set in raycasting");
    return;
  }
  // iterate over samples, multi-threaded:
#pragma omp parallel for
  for (unsigned i=0; i < particles.size(); ++i){
    Eigen::Matrix4f globalLaserOrigin;
    tf::Transform globalLaserOriginTf = particles[i].pose * base_to_laser;
    pcl_ros::transformAsMatrix(globalLaserOriginTf, globalLaserOrigin);

    // raycasting origin
    octomap::point3d originP(globalLaserOriginTf.getOrigin().x(),
                             globalLaserOriginTf.getOrigin().y(),
                             globalLaserOriginTf.getOrigin().z());
    PointCloud pc_transformed;
    pcl::transformPointCloud(pc, pc_transformed, globalLaserOrigin);

    double d = 0.0; //sum of distances for particle
    int hitCnt = 0; //number of hits from raycasting
    int missCnt = 0;
    int precCnt = 0;
    double distSqd = 0; //sum of squared distances

    // iterate over beams:
    PointCloud::const_iterator pc_it = pc_transformed.begin();
    std::vector<float>::const_iterator ranges_it = ranges.begin();
    for ( ; pc_it != pc_transformed.end(); ++pc_it, ++ranges_it){

//        if(hitCnt > 4000)
//            break;

      double p = 0.0; // probability for weight

      if (*ranges_it <= max_range){

        // direction of ray in global (map) coords
        octomap::point3d direction(pc_it->x , pc_it->y, pc_it->z);
        direction = direction - originP;

        // TODO: check first if endpoint is within map?
        octomap::point3d end;
        // raycast in OctoMap, we need to cast a little longer than max_range
        // to correct for particle drifts away from obstacles
        if(m_map->castRay(originP,direction, end, true, 1.5*max_range)){
          assert(m_map->isNodeOccupied(m_map->search(end)));
          float raycastRange = (originP - end).norm();
          float z = raycastRange - *ranges_it;
          d += fabs(z);
          distSqd += z*z;
          if(fabs(z) < 0.1)
              precCnt++;
          hitCnt++;
//          std::cout << "Distance difference is: " << z << "\t(particle " << i << ")" << std::endl;
          float sigma_scaled = m_sigmaHit;
          if (m_use_squared_error)
             sigma_scaled = (*ranges_it) * (*ranges_it) * (m_sigmaHit);

          // obstacle hit:
          p = m_zHit / (SQRT_2_PI * sigma_scaled) * exp(-(z * z) / (2 * sigma_scaled * sigma_scaled));
//          std::cout << "p after hit is: " << p << "\t(particle " << i << ")" << std::endl;

          // short range:
          if (*ranges_it <= raycastRange) {
//            p += m_zShort * m_lambdaShort * exp(-m_lambdaShort* (*ranges_it)) / (1-exp(-m_lambdaShort*raycastRange));
//            std::cout << "p after short is: " << p << "\t(particle " << i << ")" << std::endl;
          }

          // random measurement:
//          if(p < 1.0)
            p += m_zRand / max_range;
        } else { // raycasting did not hit, but measurement is no maxrange => random?
//          std::cout << "No Hit, random measurement!" << std::endl;
          p = m_zMax / max_range;
          missCnt++;
        }

      } else{ // maximum range
//          std::cout << "No Hit, maximum range!" << std::endl;
        p = m_zMax;
      }

      // add log-likelihood
      // (note: likelihood can be larger than 1!)
      assert(p > 0.0);
      particles[i].weight += log(p);

    } // end of loop over scan

    double rms = 1;
    if(hitCnt > 0){
//        rms = sqrt(distSqd/hitCnt); //without missed values
        rms = sqrt((distSqd+missSqd*missCnt)/(hitCnt+missCnt));
    }
    particles[i].rms = rms;
//    std::cout << "Particle " << i << ":" << std::endl;
//    std::cout << "Sum of distances = " << d << "\tfrom " << hitCnt <<  " hits" << std::endl;
//    std::cout << "Average distance = " << d/hitCnt << "\tfrom " << hitCnt <<  " hits" << std::endl;
//    std::cout << "Precise hits = " << precCnt << std::endl;
//    std::cout << "Missed hits = " << noHitCnt << std::endl;
//    std::cout << "Weight = " << particles[i].weight << std::endl;

  } // end of loop over particles


}

bool RaycastingModel::getHeightError(const Particle& p, const tf::StampedTransform& footprintToBase, double& heightError) const{

  octomap::point3d direction = octomap::pointTfToOctomap(footprintToBase.inverse().getOrigin());
  octomap::point3d origin = octomap::pointTfToOctomap(p.pose.getOrigin());
  octomap::point3d end;
  // cast ray to bottom:
  if (!m_map->castRay(origin, direction, end, true, 2*direction.norm()))
    return false;

  heightError =  std::max(0.0, std::abs((origin-end).z() - footprintToBase.getOrigin().z()) - m_map->getResolution());
  //ROS_INFO("Height error: %f", heightError);

  return true;
}

}
