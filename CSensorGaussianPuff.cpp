#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>
#include <chrono>
#include <functional>

#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/chrono.h>

#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Vector2d Vector2d;
typedef Eigen::VectorXd Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefMatrix;

typedef std::vector<std::vector<std::vector<int> > > vec3d;
typedef std::chrono::system_clock::time_point TimePoint;

class CSensorGaussianPuff{

public:

    int sim_dt, puff_dt;
    int puff_duration;

    int N_sensors;
    const Vector X, Y, Z;
    const Vector wind_speeds, wind_directions;
    Vector X_rot, Y_rot;
    Vector sigma_y, sigma_z;
    time_t sim_start, sim_end;
    Matrix source_coordinates;
    Vector emission_strengths;
    double x0, y0, z0; // source coordintaes for curent source
    Matrix s; // shifted sensor coordinates for current source
    double x_min, y_min; // current mins for the grid centered at the current source
    double x_max, y_max; 

    std::function<double(double)> exp;

    double conversion_factor;
    double exp_tol;

    bool quiet;

    const double two_pi_three_halves = std::pow(2*M_PI, 1.5);
    double cosine; // store value of cosine/sine so we don't have to evaluate it across different functions
    double sine;

    /* Constructor.
    Inputs:
        X, Y, Z: Flattened versions of 3D meshgrids
        nx, ny, nz: number of points in each direction
        sim_dt: time between simulation time steps (MUST BE INTEGER NUMBER OF SECONDS)
        puff_dt: time between creation of two puffs
        puff_duration: maximum number of seconds puff can be live for. The default is no threshold, since the code
            computes the correct cutoff time. Only add a puff duration if you need faster runtimes at the expense of accuracy.
        sim_start, sim_end: datetime stamps for when to start and end the emission
        wind_speeds, wind_directions: timeseries for wind speeds (m/s) and directions (degrees) at sim_dt resolution
        source_coordinates: source coordinates in (x,y,z) format for each source. size- (n_sources, 3)
        emission_strengths: emission rates for each source (kg/hr). length: n_sources
        conversion_factor: conversion between kg/m^3 to ppm for ch4
        exp_tol: tolerance for the exponential thresholding applied to the Gaussians. Lower tolerance means less accuracy
        but a faster execution. Runtime and accuracy are both very sensitive to this parameter.
        quiet: false if you want output for simulation completeness. true for silent simulation.
    */
    CSensorGaussianPuff(Vector X, Vector Y, Vector Z, 
                    int N_sensors,
                    int sim_dt, int puff_dt, int puff_duration,
                    TimePoint sim_start, TimePoint sim_end,
                    Vector wind_speeds, Vector wind_directions,
                    Matrix source_coordinates, Vector emission_strengths,
                    double conversion_factor, double exp_tol,
                    bool unsafe, bool quiet)
    : X(X), Y(Y), Z(Z), N_sensors(N_sensors),
    sim_dt(sim_dt), puff_dt(puff_dt), puff_duration(puff_duration), wind_speeds(wind_speeds), wind_directions(wind_directions),
    source_coordinates(source_coordinates), emission_strengths(emission_strengths),
    conversion_factor(conversion_factor), exp_tol(exp_tol), quiet(quiet) {

        s.resize(2, X.size());

        if(unsafe){
            if (!quiet) std::cout << "RUNNING IN UNSAFE MODE\n";
            this->exp = &fastExp; 
        } else {
            this->exp = [](double x){return std::exp(x);};
        }


        sigma_y = Vector(N_sensors);
        sigma_z = Vector(N_sensors);

        this->sim_start = std::chrono::system_clock::to_time_t(sim_start);
        this->sim_end = std::chrono::system_clock::to_time_t(sim_end);
    }

    /* Rotates the X and Y grids based on the current wind direction and source location.
    Inputs:
        X_rot, Y_rot: vectors the same size as the grid to get filled with rotated coordinates.
    Returns:
        None, but fills X_rot and Y_rot with the rotated grids.
    */
    void rotateSensors(RefVector X_rot, RefVector Y_rot){

        Eigen::Matrix2d R;
        R << cosine, -sine,
            sine, cosine;

        Matrix R_s = R*s;

        X_rot = R_s.row(0);
        Y_rot = R_s.row(1);
    }

    /* Axis Aligned Bounding Box algorithm. Used to compute the intersections between a ray (wind direction) and a square.
    See https://tavianator.com/2022/ray_box_boundary.html for details.
    Inputs:
        box_min, box_max: 2D vectors containing the minimum and maximum corners of a square in the xy plane.
        origin: starting location of the ray.
        invRayDir: elementwise inverse of the ray direction. While the ray direction could be used instead as the input,
            using the inverse saves on computing multiple divisions.
    Returns:
        2D vector containing the times of the ray intersection. For the fast Gaussian Puff algorithm, the ray's origin
        is always within the box. As such, the returns will have tmin < 0 be the backwards intersection and tmax > 0 
        be the forward intersection, where the directions refer to traveling in the positive and negative ray direction.
    */
    Vector AABB(Vector box_min, Vector box_max, Vector origin, Vector invRayDir){

        // casts to arrays make it an elementwise product
        Vector t0 = (box_min-origin).array()*invRayDir.array();
        Vector t1 = (box_max-origin).array()*invRayDir.array();

        double tmax = (t0.cwiseMax(t1)).minCoeff();
        double tmin = (t0.cwiseMin(t1)).maxCoeff();

        return Vector2d(tmin, tmax);
    }
 
    /*  Given a point on the edge of a square, finds the nearest of the four corners of the square.
    Inputs:
        min_corner, max_corner:  minimum and maximum corners of a square (e.g. lower left and top right).
        point: a point on the edge of the square.
    Returns:
        2D Vector containing the coordintes to the nearest corner of the square. 
    */
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

    /* Computes Pasquill stability class
    Inputs:
        wind_speed: [m/s]
        hour: current hour of day
    Returns:
        stability_class: character A-F representing a Pasquill stability class
    */
    char stabilityClassifier(double wind_speed, int hour, int day_start=7, int day_end=19) {
        bool is_day = (hour >= day_start) && (hour <= day_end);
        char stability_class;

        if (wind_speed < 2.0) {
            stability_class = is_day ? 'A' : 'E';
        } else if (wind_speed < 5.0) {
            stability_class = is_day ? 'B' : 'E';
        } else {
            stability_class = 'D';
        }

        return stability_class;
    }

    /* Gets dispersion coefficients (sigma_{y,z}) for the entire grid.
        sigma_z = a*x^b, x in km,
        sigma_y = 465.11628*x*tan(THETA) where THETA = 0.017453293*(c-d*ln(x)) where x in km
        Note: sigma_{y,z} = -1 if x < 0 due to there being no upwind dispersion.
    Inputs:
        stability_class: a char in A-F from Pasquill stability classes 
        X_rot: rotated version of the X grid
    Returns:
        None, but sigma_y and sigma_z class variables are filled with the dispersion coefficients.
    */
    void getSigmaCoefficients(char stability_class, Vector X_rot){
        X_rot = X_rot.array() * 0.001; // convert to km

        for(int i = 0; i < X_rot.size(); i++){
            int flag = 0;
            double a, b, c, d;

            double x = X_rot[i];

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


    double calculatePlumeTravelTime(double thresh_xy, double ws){

        // max sized plume centered on source
        Vector2d box_min(-thresh_xy, -thresh_xy);
        Vector2d box_max(thresh_xy, thresh_xy);

        Vector2d grid_min(x_min, y_min); // implicit grid based on sensor positions
        Vector2d grid_max(x_max, y_max);

        Vector2d origin(0,0);

        Vector2d rayDir(cosine, -sine);
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
        double travelDistance = (distance.array()*invRayDir.array()).minCoeff();
        double travelTime = travelDistance/ws;

        return travelTime;
    }

    void GaussianPuffEquation(
        double q, double ws,
        RefVector X_rot, RefVector Y_rot,
        RefMatrix ch4){

        double sigma_y_max = sigma_y.maxCoeff();
        double sigma_z_max = sigma_z.maxCoeff();

        // compute thresholds
        double prefactor = (q * conversion_factor) / (two_pi_three_halves * sigma_y_max*sigma_y_max * sigma_z_max);
        double threshold = std::log(exp_tol / (2*prefactor));
        double thresh_constant = std::sqrt(-2*threshold);

        double thresh_xy_max = sigma_y_max*thresh_constant;
        double thresh_z_max = sigma_z_max*thresh_constant;

        double t = calculatePlumeTravelTime(thresh_xy_max, ws); // number of seconds til plume leaves grid

        int n_time_steps = ceil(t/sim_dt);

        // bound check on time
        if(n_time_steps >= ch4.rows()){
            n_time_steps = ch4.rows() - 1;
        }

        for(int i = 0; i <= n_time_steps; i++){

            double wind_shift = ws*(i*sim_dt); // i*sim_dt is # of seconds on current time step
            Vector X_rot_shift = X_rot.array() - wind_shift; // wind shift

            for(int j = 0; j < N_sensors; j++){

                // Skips upwind sensors since sigma_{y,z} = -1 for upwind points
                if (sigma_y[j] < 0 || sigma_z[j] < 0) {
                    continue;
                }

                double t_xy = sigma_y[j]*thresh_constant; // local threshold

                // Exponential thresholding conditionals
                if (std::abs(X_rot_shift[j]) >= t_xy) {
                    continue;
                }

                if (std::abs(Y_rot[j]) >= t_xy) {
                    continue;
                }

                double t_z = sigma_z[j]*thresh_constant; // local threshold

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
                double term_4 = this->exp(-0.5*(term_3_arg + term_4_a_arg)) + this->exp(-0.5*(term_3_arg + term_4_b_arg));

                ch4(i, j) += term_1 * term_4 * conversion_factor;

            }
        }
    }

    /* Computes the concentration timeseries for a single puff.
    Inputs:
        q: Total emission corresponding to this puff (kg)
        ws, wind speed (m/s)
        hour: current hour of day (int)
        ch4: 2D concentration array. First index is time, second index is the flattened spatial index.
    Returns:
        None. The concentration is added directly into the ch4 array in GaussianPuffEquation()
    */
    void concentrationPerPuff(double q, double theta, double ws, int hour,
                                RefMatrix ch4){

        // cache cos/sin so they can get reused in other calls
        cosine = cos(theta);
        sine = sin(theta);

        Vector X_rot(X.size());
        Vector Y_rot(Y.size());

        // rotates X and Y grids, stores in X_rot and Y_rot
        rotateSensors(X_rot, Y_rot);

        char stability_class = stabilityClassifier(ws, hour);

        // gets sigma coefficients and stores in sigma_{y,z} class member vars
        getSigmaCoefficients(stability_class, X_rot);

        GaussianPuffEquation(q, ws,
                                X_rot, Y_rot,
                                ch4);
    }

    /* Simulation time loop
    Inputs:
        ch4: 2d array. First index represents simulation time steps, second index is flattened spatial index
    Returns:
        none, but the concentration is added directly to the ch4 array for all time steps
    */
    void simulate(RefMatrix ch4){

        double emission_length = difftime(sim_end, sim_start);
        int n_puffs = ceil(emission_length/puff_dt);

        // later, for multisource: iterate over source coords
        setSourceCoordinates(0);
        double q = emission_strengths[0]/3600; // convert to kg/s

        double emission_per_puff = q*puff_dt;

        time_t current_time = sim_start;
        double report_ratio = 0.1;

        int puff_lifetime = ceil(puff_duration/sim_dt);
        int ratio = puff_dt/sim_dt;

        for(int t = 0; t < n_puffs; t++){
            
            // keeps track of current time. needed to compute stability class
            tm puff_start = *localtime(&current_time);
            current_time += puff_dt;

            // bounds check on time
            if(t*ratio + puff_lifetime >= ch4.rows()) puff_lifetime = ch4.rows()-t*ratio;

            double theta = windDirectionToAngle(wind_directions[t]);

            // computes concentration timeseries for this puff
            concentrationPerPuff(emission_per_puff, theta, wind_speeds[t], 
                                    puff_start.tm_hour, ch4.middleRows(t*ratio, puff_lifetime));
            
            if(!quiet && floor(n_puffs*report_ratio) == t){
                std::cout << "Simulation is " << report_ratio*100 << "\% done\n";
                report_ratio += 0.1;
            }
        }
    }

private:

    static double fastExp(double x){
        constexpr double a = (1ll << 52) / 0.6931471805599453;
        constexpr double b = (1ll << 52) * (1023 - 0.04367744890362246);
        x = a * x + b;

        constexpr double c = (1ll << 52);
        if (x < c)
            x = 0.0;

        uint64_t n = static_cast<uint64_t>(x);
        std::memcpy(&x, &n, 8);
        return x;
    }

    void setSourceCoordinates(int source_index){
        x0 = source_coordinates(source_index, 0);
        y0 = source_coordinates(source_index, 1);
        z0 = source_coordinates(source_index, 2);

        Vector X_shift = X.array() - x0;
        Vector Y_shift = Y.array() - y0;

        s << X_shift.transpose(), Y_shift.transpose();

        x_min = X.minCoeff() - x0;
        y_min = Y.minCoeff() - y0;
        x_max = X.maxCoeff() - x0;
        y_max = Y.maxCoeff() - y0;
    }

    // convert wind direction (degrees) to the angle (radians) between the wind vector and positive x-axis
    double windDirectionToAngle(double wd){
        double theta = wd-270;
        if(theta < 0) theta = theta + 360;
        theta = theta*(M_PI/180.0); // convert to radians

        return theta;
    }
};


using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(CSensorGaussianPuff, m) {
    // m.doc() = "Gaussian Puff code";

    py::class_<CSensorGaussianPuff>(m, "CSensorGaussianPuff")
    .def(py::init<Vector, Vector, Vector, int, int, int, int, 
                    TimePoint, TimePoint, 
                    Vector, Vector, Matrix, Vector, double, double, bool, bool>())
    .def("simulate", &CSensorGaussianPuff::simulate);
}