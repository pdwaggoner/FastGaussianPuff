#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>

#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Vector2d Vector2d;
typedef Eigen::VectorXd Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefMatrix;

typedef std::vector<std::vector<std::vector<int> > > vec3d;


struct timeval tv;
struct timezone tz;
double timeNow() {
    gettimeofday( &tv, &tz );
    int _mils = tv.tv_usec/1000;
    int _secs = tv.tv_sec;
    return (double)_secs + ((double)_mils/1000.0);
}

class CGaussianPuff{

public:

    const Vector X, Y, Z;
    Vector X_rot, Y_rot;
    Vector sigma_y, sigma_z;
    int nx, ny, nz;
    double dx, dy, dz;
    double x_min, y_min;

    double conversion_factor;
    double exp_tol;

    const double two_pi_three_halves = std::pow(2*M_PI, 1.5);

    vec3d map_table;

    CGaussianPuff(Vector X, Vector Y, Vector Z, 
                    int nx, int ny, int nz, 
                    double conversion_factor, double exp_tol,
                    vec3d map_table)

    : X(X), Y(Y), Z(Z) , nx(nx), ny(ny), nz(nz), 
    conversion_factor(conversion_factor), exp_tol(exp_tol) {

        std::vector<double> gridSpacing = computeGridSpacing();
        dx = gridSpacing[0];
        dy = gridSpacing[1];
        dz = gridSpacing[2];

        sigma_y = Vector(nx*ny*nz);
        sigma_z = Vector(nx*ny*nz);

        // precomputes the map from the 3D meshgrid index to the 1D raveled index
        for(int i = 0; i < nx; i++){
            for(int j = 0; j < ny; j++){
                for(int k = 0; k < nz; k++){
                    // (i,j) index flipped since numpy's 'ij' indexing is being used on the meshgrids
                    map_table[j][i][k] = map(i,j,k);
                }
            }
        }
        this->map_table = map_table;
    }

    std::vector<double> computeIndexBounds(double thresh_xy, double thresh_z,
                                            double ws, double wd, double t_i,
                                            double x0, double y0, double z0){

        double x_min = X.minCoeff() - x0;
        double y_min = Y.minCoeff() - y0;

        double theta = windDirectionToAngle(wd);

        Eigen::Matrix2d R;
        R << cos(theta), sin(theta),
            -sin(theta), cos(theta);

        Eigen::Vector2d X0;
        X0 << x_min, y_min;

        Eigen::Vector2d v;
        v << cos(theta), -sin(theta);

        Eigen::Vector2d vp;
        vp << sin(theta), cos(theta);

        Eigen::Vector2d tw;
        tw << t_i*ws, 0;

        Eigen::Vector2d X0_r = R*X0;
        auto X0_rt = X0_r - tw;

        double Xrt_dot_v = X0_rt.dot(v);
        double Xrt_dot_vp = X0_rt.dot(vp);
        double norm_sq_Xrt = X0_rt.dot(X0_rt);

        double i_lower = (-Xrt_dot_v - thresh_xy - 1)/dx;
        double i_upper = (-Xrt_dot_v + thresh_xy + 1)/dx;

        double j_lower = (-Xrt_dot_vp - thresh_xy - 1)/dy;
        double j_upper = (-Xrt_dot_vp + thresh_xy + 1)/dy;

        double k_lower = (-thresh_z + z0)/dz;
        double k_upper = (thresh_z + z0)/dz;

        return std::vector<double>{i_lower, i_upper, j_lower, j_upper, k_lower, k_upper};
    }

    std::vector<int> getValidIndices(double thresh_xy, double thresh_z,
                                        double ws, double wd, double t_i,
                                        double x0, double y0, double z0){
        
        std::vector<double> indexBounds = computeIndexBounds(thresh_xy, thresh_z,
                                                            ws, wd, t_i,
                                                            x0, y0, z0);

        int i_lower = floor(indexBounds[0]);
        int i_upper = ceil(indexBounds[1]);
        int j_lower = floor(indexBounds[2]);
        int j_upper = ceil(indexBounds[3]);
        int k_lower = floor(indexBounds[4]);
        int k_upper = ceil(indexBounds[5]);

        // makes sure the computed index bounds are sensical and computes total number of cells in bounds
        if(i_lower < 0) i_lower = 0;
        if(i_upper > nx-1) i_upper = nx-1;

        if(j_lower < 0) j_lower = 0;
        if(j_upper > ny-1) j_upper = ny-1;

        if(k_lower < 0) k_lower = 0;
        if(k_upper > nz-1) k_upper = nz-1;

        int i_count, j_count, k_count;
        if(i_upper < i_lower || i_lower > i_upper){
            return std::vector<int>(0);
        } else{
            i_count = i_upper-i_lower+1;
        }

        if(j_upper < j_lower || j_lower > j_upper){
            return std::vector<int>(0);
        } else{
            j_count = j_upper-j_lower+1;
        }

        if(k_upper < k_lower){
            return std::vector<int>(0);
        } else{
            k_count = k_upper-k_lower+1;
        }

        int cellCount = i_count*j_count*k_count;

        std::vector<int> indices(cellCount);
        int currentCell = 0;
        for(int i = i_lower; i <= i_upper; i++){
            for(int j = j_lower; j <= j_upper; j++){
                for(int k = k_lower; k <= k_upper; k++){
                    indices[currentCell] = map_table[j][i][k];
                    currentCell++;
                }
            }
        }

        return indices;
    }

    void rotateGrids(double x0, double y0, double z0, double wd,
                                    RefVector X_rot, RefVector Y_rot){

        double x_min = X.minCoeff() - x0;
        double y_min = Y.minCoeff() - y0;

        double theta = windDirectionToAngle(wd);

        Eigen::Matrix2d R;
        R << cos(theta), sin(theta),
            -sin(theta), cos(theta);

        Eigen::Vector2d X0;
        X0 << x_min, y_min;

        Eigen::Vector2d v;
        v << cos(theta), -sin(theta);

        Eigen::Vector2d vp;
        vp << sin(theta), cos(theta);

        Eigen::Vector2d X_r = R*X0;

        // shifted to be centered at the source then bumped up to start at 0 to align with indexing
        Vector X_shift = X.array()-x0-x_min;
        Vector Y_shift = Y.array()-y0-y_min;

        Eigen::Matrix2Xd rotated(2, X_shift.size());

        X_rot = X_r[0] + X_shift.array()*v[0] + Y_shift.array()*vp[0];
        Y_rot = X_r[1] + X_shift.array()*v[1] + Y_shift.array()*vp[1];
    }

    Vector AABB(Vector box_min, Vector box_max, Vector origin, Vector invRayDir){

        // casts to arrays make it an elementwise product
        Vector t0 = (box_min-origin).array()*invRayDir.array();
        Vector t1 = (box_max-origin).array()*invRayDir.array();

        double tmax = (t0.cwiseMax(t1)).minCoeff();
        double tmin = (t0.cwiseMin(t1)).maxCoeff();

        return Vector2d(tmin, tmax);
    }

    Vector findNearestCorner(Vector min_corner, Vector max_corner, Vector point){
        
        Vector2d corner;

        if(abs(min_corner(0)-point(0)) < abs(max_corner(0)-point(0))){
            corner(0) = min_corner(0);
        }else{
            corner(0) = max_corner(0);
        }

        if(abs(min_corner(1)-point(1)) < abs(max_corner(1)-point(1))){
            corner(1) = min_corner(1);
        }else{
            corner(1) = max_corner(1);
        }

        return corner;
    }

    // should we rewrite the AABB calls from here directly into this function?
    double calculatePlumeTravelTime(double thresh_xy, 
                                    double ws, double wd, 
                                    double x0, double y0){

        // doesn't need z parameters since plume moves in 2D
        std::vector<double> start_box = computeIndexBounds(thresh_xy, 0, 
                                        ws, wd, 0, 
                                        x0, y0, 0);

        double i_min = start_box[0];
        double i_max = start_box[1];
        double j_min = start_box[2];
        double j_max = start_box[3];

        double x_min = X.minCoeff() - x0;
        double y_min = Y.minCoeff() - y0;
        double x_max = X.maxCoeff() - x0;
        double y_max = Y.maxCoeff() - y0;

        // corners of the threshold box
        double box_min_x = x_min + i_min*dx;
        double box_min_y = y_min + j_min*dy;
        double box_max_x = x_min + i_max*dx;
        double box_max_y = y_min + j_max*dy;

        Vector2d box_min(box_min_x, box_min_y);
        Vector2d box_max(box_max_x, box_max_y);

        Vector2d grid_min(x_min, y_min);
        Vector2d grid_max(x_max, y_max);

        Vector2d origin(0,0);

        double theta = windDirectionToAngle(wd);
        Vector2d rayDir(cos(theta), sin(theta));
        Vector2d invRayDir = rayDir.cwiseInverse();

        // finding the last corner of the threshold box to leave the grid
        Vector2d box_times = AABB(box_min, box_max, origin, invRayDir); // find where ray intersects box
        Vector2d backward_collision = box_times[0]*rayDir; // where backwards ray intersects with an edge of the box
        Vector2d box_corner = findNearestCorner(box_min, box_max, backward_collision);

        // find the corner of the grid that the threshold must pass based on the wind direction
        Vector2d grid_middle = (grid_max-grid_min).array()/2 + grid_min.array();
        Vector2d grid_times = AABB(grid_min, grid_max, grid_middle, invRayDir);
        Vector2d forward_collision = grid_times[1]*rayDir + grid_middle;
        Vector2d grid_corner = findNearestCorner(grid_min, grid_max, forward_collision);

        // compute travel time between the two corners
        Vector2d distance = (grid_corner-box_corner).cwiseAbs();
        invRayDir = invRayDir.cwiseAbs();
        double travelTime = (distance.array()*invRayDir.array()).minCoeff();

        return travelTime;
    }

    void getSigmaCoefficients(char stability_class, Vector XY){
        XY = XY.array() * 0.001; // convert to km

        for(int i = 0; i < XY.size(); i++){
            int flag = 0;
            double a, b, c, d;

            double x = XY[i];

            if (x <= 0) {
                sigma_y[i] = -1;
                sigma_z[i] = -1;
            } else {
                if (stability_class == 'A') {
                    if (x < 0.1) {
                        a = 122.800;
                        b = 0.94470;
                    } else if (x >= 0.1 && x < 0.15) {
                        a = 158.080;
                        b = 1.05420;
                    } else if (x >= 0.15 && x < 0.20) {
                        a = 170.220;
                        b = 1.09320;
                    } else if (x >= 0.20 && x < 0.25) {
                        a = 179.520;
                        b = 1.12620;
                    } else if (x >= 0.25 && x < 0.30) {
                        a = 217.410;
                        b = 1.26440;
                    } else if (x >= 0.30 && x < 0.40) {
                        a = 258.890;
                        b = 1.40940;
                    } else if (x >= 0.40 && x < 0.50) {
                        a = 346.750;
                        b = 1.72830;
                    } else if (x >= 0.50 && x < 3.11) {
                        a = 453.850;
                        b = 2.11660;
                    } else {
                        flag = 1;
                    }
                    c = 24.1670;
                    d = 2.5334;
                } else if (stability_class == 'B') {
                    if (x < 0.2) {
                        a = 90.673;
                        b = 0.93198;
                    } else if (x >= 0.2 && x < 0.4) {
                        a = 98.483;
                        b = 0.98332;
                    } else {
                        a = 109.300;
                        b = 1.09710;
                    }
                    c = 18.3330;
                    d = 1.8096;
                } else if (stability_class == 'C') {
                    a = 61.141;
                    b = 0.91465;
                    c = 12.5000;
                    d = 1.0857;
                } else if (stability_class == 'D') {
                    if (x < 0.3) {
                        a = 34.459;
                        b = 0.86974;
                    } else if (x >= 0.3 && x < 1) {
                        a = 32.093;
                        b = 0.81066;
                    } else if (x >= 1 && x < 3) {
                        a = 32.093;
                        b = 0.64403;
                    } else if (x >= 3 && x < 10) {
                        a = 33.504;
                        b = 0.60486;
                    } else if (x >= 10 && x < 30) {
                        a = 36.650;
                        b = 0.56589;
                    } else {
                        a = 44.053;
                        b = 0.51179;
                    }
                    c = 8.3330;
                    d = 0.72382;
                } else if (stability_class == 'E') {
                    if (x < 0.1) {
                        a = 24.260;
                        b = 0.83660;
                    } else if (x >= 0.1 && x < 0.3) {
                        a = 23.331;
                        b = 0.81956;
                    } else if (x >= 0.3 && x < 1) {
                        a = 21.628;
                        b = 0.75660;
                    } else if (x >= 1 && x < 2) {
                        a = 21.628;
                        b = 0.63077;
                    } else if (x >= 2 && x < 4) {
                        a = 22.534;
                        b = 0.57154;
                    } else if (x >= 4 && x < 10) {
                        a = 24.703;
                        b = 0.50527;
                    } else if (x >= 10 && x < 20) {
                        a = 26.970;
                        b = 0.46173;
                    } else if (x >= 20 && x < 40) {
                        a = 35.420;
                        b = 0.37615;
                    } else {
                        a = 47.618;
                        b = 0.29592;
                    }
                    c = 6.2500;
                    d = 0.54287;
                } else if (stability_class == 'F') {
                    if (x < 0.2) {
                        a = 15.209;
                        b = 0.81558;
                    } else if (x >= 0.2 && x < 0.7) {
                        a = 14.457;
                        b = 0.78407;
                    } else if (x >= 0.7 && x < 1) {
                        a = 13.953;
                        b = 0.68465;
                    } else if (x >= 1 && x < 2) {
                        a = 13.953;
                        b = 0.63227;
                    } else if (x >= 2 && x < 3) {
                        a = 14.823;
                        b = 0.54503;
                    } else if (x >= 3 && x < 7) {
                        a = 16.187;
                        b = 0.46490;
                    } else if (x >= 7 && x < 15) {
                        a = 17.836;
                        b = 0.41507;
                    } else if (x >= 15 && x < 30) {
                        a = 22.651;
                        b = 0.32681;
                    } else if (x >= 30 && x < 60) {
                        a = 27.074;
                        b = 0.27436;
                    } else {
                        a = 34.219;
                        b = 0.21716;
                    }
                    c = 4.1667;
                    d = 0.36191;
                } else {
                    throw std::invalid_argument("Invalid stability class.");
                }

                if (flag == 0) {
                    double Theta = 0.017453293 * (c - d * std::log(x)); // in radians
                    sigma_y[i] = 465.11628 * x * std::tan(Theta); // in meters
                    sigma_z[i] = a * std::pow(x, b); // in meters
                    sigma_z[i] = std::min(sigma_z[i], 5000.0);
                } else {
                    sigma_z[i] = 5000.0;
                }
            }
        }
    }

    void GaussianPuffEquation(
        double q, double ws, double wd,
        double x0, double y0, double z0,
        RefVector X_rot, RefVector Y_rot,
        const RefVector ts, RefMatrix c){

        double sigma_y_max = sigma_y.maxCoeff();
        double sigma_z_max = sigma_z.maxCoeff();

        // THRESHOLD STUFF
        double term_1_thresh = q / (two_pi_three_halves * std::pow(sigma_y_max, 2) * sigma_z_max);
        double emission_strength = term_1_thresh * conversion_factor; // called q_{xy} in the writeup
        double threshold = std::log(exp_tol / emission_strength);
        double thresh_constant = std::sqrt(-2*threshold);

        double thresh_xy_max = sigma_y_max*thresh_constant;
        double thresh_z_max = sigma_z_max*thresh_constant;

        double t_raw = calculatePlumeTravelTime(thresh_xy_max, ws, wd, x0, y0);
        int t = ceil(t_raw/ws);

        // bound check on time
        if(t >= ts.size()){
            t = ts.size()-1;
        }


        for (int i = t; i >= 0; i--) {

            std::vector<int> indices = getValidIndices(thresh_xy_max, thresh_z_max, 
                                        ws, wd, ts[i], 
                                        x0, y0, z0);
            

            if(indices.empty()){
                continue;
            }

            // shrinks the thresholds
            double box_max_sig_y = sigma_y(indices).maxCoeff(); // max sigma of the current valid indices
            double box_max_sig_z = sigma_z(indices).maxCoeff();
            thresh_xy_max = box_max_sig_y*thresh_constant;
            thresh_z_max = box_max_sig_z*thresh_constant;

            Vector X_rot_shift = X_rot.array() - ts[i]*ws; // wind shift

            for (int j : indices) {

                // Skips upwind grid cells since sigma_{y,z} = -1 for upwind points
                if (sigma_y[j] < 0 || sigma_z[j] < 0) {
                    continue;
                }

                double t_xy = sigma_y[j]*thresh_constant;

                // Exponential thresholding conditionals
                if (std::abs(X_rot_shift[j]) >= t_xy) {
                    continue;
                }

                if (std::abs(Y_rot[j]) >= t_xy) {
                    continue;
                }

                double t_z = sigma_z[j]*thresh_constant;

                if (std::abs(Z[j] - z0) >= t_z) {
                    continue;
                }

                // terms are written in a way to minimize divisions and exp evaluations
                double one_over_sig_y = 1/sigma_y[j];
                double one_over_sig_z = 1/sigma_z[j];

                double y_by_sig = Y_rot[j] * one_over_sig_y;
                double x_by_sig = X_rot_shift[j] * one_over_sig_y;
                double z_minus_by_sig = (Z[j] - z0) * one_over_sig_z;
                double z_plus_by_sig = (Z[j] + z0) * one_over_sig_z;

                double term_4_a_arg = z_minus_by_sig*z_minus_by_sig;
                double term_4_b_arg = z_plus_by_sig*z_plus_by_sig;
                double term_3_arg = (y_by_sig*y_by_sig + x_by_sig*x_by_sig);

                double term_1 = q / (two_pi_three_halves * sigma_y[j]*sigma_y[j] * sigma_z[j]);
                double term_4 = std::exp(-0.5*(term_3_arg + term_4_a_arg)) + std::exp(-0.5*(term_3_arg + term_4_b_arg));

                c(i, j) += term_1 * term_4 * conversion_factor;
            }
        }
    }

    void concentrationPerPuff(double q, double wd, double ws, 
                                double x0, double y0, double z0, 
                                double hour, char stability_class,
                                RefVector& times, RefMatrix& ch4){

        Vector X_rot(X.size());
        Vector Y_rot(Y.size());

        // rotates X and Y grids, stores in X_rot and Y_rot
        rotateGrids(x0, y0, z0, wd, X_rot, Y_rot);

        // gets sigma coefficients and stores in sigma_{y,z} member vars
        getSigmaCoefficients(stability_class, X_rot);

        GaussianPuffEquation(q, ws, wd,
                                x0, y0, z0,
                                X_rot, Y_rot,
                                times, ch4);

    }

private:

    double windDirectionToAngle(double wd){
        double theta = 270 - wd;
        if(theta < 0) theta = theta + 360;
        theta = theta*(M_PI/180.0); // convert to radians

        return theta;
    }

    std::vector<double> computeGridSpacing(){

        std::vector<double> gridSpacing(3); 

        gridSpacing[0] = abs(X[nz] - X[0]); // dx
        gridSpacing[1] = abs(Y[nz*nx] - Y[0]); // dy
        gridSpacing[2] = abs(Z[1] - Z[0]); // dz

        return gridSpacing;
    }

    // maps 3d index to 1d raveled index in numpy 'xy' format meshgrids
    int map(int i, int j, int k){
        return j*nz*nx + i*nz + k;
    }

    // inverts a 1d raveled numpy index to a 3d index in numpy 'xy' format
    std::vector<int> invertIndex(int l){

        int j = int(l / (nx * nz));
        l -= (j * nx * nz);
        int i = floor(l / nz);
        int k = l % nz;

        return std::vector<int>{i,j,k};
    }
    
    // temp testing function
    void invertIdxArray(std::vector<int> idx_arr, int nx, int ny, int nz){

        int imin = nx;  // Initialize minimum and maximum indices
        int imax = -1;
        int jmin = ny;
        int jmax = -1;
        int kmin = nz;
        int kmax = -1;

        if(idx_arr.empty()){
            std::cout << "INDEX ARRAY EMPTY\n";
            return;
        }

        for(int l : idx_arr){
            std::vector<int> idx = invertIndex(l);
            int i = idx[0];
            int j = idx[1];
            int k = idx[2];

            if(i<imin) imin = i;
            if(i>imax) imax = i;
            if(j<jmin) jmin = j;
            if(j>jmax) jmax = j;
            if(k<kmin) kmin = k;
            if(k>kmax) kmax = k;

        }

        std::cout << imin << " <= i <= " << imax << std::endl;
        std::cout << jmin << " <= j <= " << jmax << std::endl;
        std::cout << kmin << " <= k <= " << kmax << std::endl;
    }

};


using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(CGaussianPuff, m) {
    // m.doc() = "Gaussian Puff code";

    py::class_<CGaussianPuff>(m, "CGaussianPuff")
    .def(py::init<Vector, Vector, Vector, int, int, int, double, double, vec3d>())
    .def("GaussianPuffEquation", &CGaussianPuff::GaussianPuffEquation)
    .def("rotateGrids", &CGaussianPuff::rotateGrids)
    .def("concentrationPerPuff", &CGaussianPuff::concentrationPerPuff)
    .def("getSigmaCoefficients", &CGaussianPuff::getSigmaCoefficients);

    // py::class_<CGaussianPuff>(m, "CGaussianPuff").def(py::init());
    // def("GaussianPuffEquation", &GaussianPuffEquation, "evaluate gaussian puff equation"),
    // def("getDownwindCoordinates", &get_downwind_coordinates, "gets rotated grids");
}