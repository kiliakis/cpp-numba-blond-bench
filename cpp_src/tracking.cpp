/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/


#include <string.h>     // memset()
#include <stdlib.h>     // mmalloc()
#include <cmath>
#include "openmp.h"
#include "sin.h"

using namespace vdt;

extern "C" void linear_interp_kick(double * __restrict__ beam_dt,
                                   double * __restrict__ beam_dE,
                                   const double * __restrict__ voltage_array,
                                   const double * __restrict__ bin_centers,
                                   const double charge,
                                   const int n_slices,
                                   const int n_macroparticles,
                                   const double acc_kick)
{


    const int STEP = 64;
    const double inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1]
                                    - bin_centers[0]);

    double *voltageKick = (double *) malloc ((n_slices - 1) * sizeof(double));
    double *factor = (double *) malloc ((n_slices - 1) * sizeof(double));

    #pragma omp parallel
    {
        unsigned fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_slices - 1; i++) {
            voltageKick[i] =  charge * (voltage_array[i + 1] - voltage_array[i]) * inv_bin_width;
            factor[i] = (charge * voltage_array[i] - bin_centers[i] * voltageKick[i]) + acc_kick;
        }

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            for (int j = 0; j < loop_count; j++) {
                fbin[j] = (unsigned) std::floor((beam_dt[i + j] - bin_centers[0])
                                                * inv_bin_width);
            }

            for (int j = 0; j < loop_count; j++) {
                if (fbin[j] < n_slices - 1) {
                    beam_dE[i + j] += beam_dt[i + j] * voltageKick[fbin[j]] + factor[fbin[j]];
                }
            }

        }
    }
    free(voltageKick);
    free(factor);
}



extern "C" void kick(const double * __restrict__ beam_dt, 
                     double * __restrict__ beam_dE, const int n_rf, 
                     const double * __restrict__ voltage, 
                     const double * __restrict__ omega_RF, 
                     const double * __restrict__ phi_RF,
                     const int n_macroparticles,
                     const double acc_kick){
int j;

// KICK
for (j = 0; j < n_rf; j++)
#pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
                beam_dE[i] = beam_dE[i] + voltage[j]
                           * fast_sin(omega_RF[j] * beam_dt[i] + phi_RF[j]);

// SYNCHRONOUS ENERGY CHANGE
#pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] = beam_dE[i] + acc_kick;

}

extern "C" void rf_volt_comp(const double * __restrict__ voltage,
                             const double * __restrict__ omega_RF,
                             const double * __restrict__ phi_RF,
                             const double * __restrict__ bin_centers,
                             const int n_rf,
                             const int n_bins,
                             double *__restrict__ rf_voltage)
{
    for (int j = 0; j < n_rf; j++) {
        #pragma omp parallel for
        for (int i = 0; i < n_bins; i++) {
            rf_voltage[i] += voltage[j]
                             * fast_sin(omega_RF[j] * bin_centers[i] + phi_RF[j]);
        }
    }
}



extern "C" void histogram(const double *__restrict__ input,
                          double *__restrict__ output, const double cut_left,
                          const double cut_right, const int n_slices,
                          const int n_macroparticles)
{
    // Number of Iterations of the inner loop
    const int STEP = 16;
    const double inv_bin_width = n_slices / (cut_right - cut_left);

    // allocate memory for the thread_private histogram
    double **histo = (double **) malloc(omp_get_max_threads() * sizeof(double *));
    histo[0] = (double *) malloc (omp_get_max_threads() * n_slices * sizeof(double));
    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0., n_slices * sizeof(double));
        float fbin[STEP];
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            // First calculate the index to update
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = floor((input[i + j] - cut_left) * inv_bin_width);
            }
            // Then update the corresponding bins
            for (int j = 0; j < loop_count; j++) {
                const int bin  = (int) fbin[j];
                if (bin < 0 || bin >= n_slices) continue;
                histo[id][bin] += 1;
            }
        }

        // Reduce to a single histogram
        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }
    }

    // free memory
    free(histo[0]);
    free(histo);
}


extern "C" void histogram_v2(const double *__restrict__ input,
                          double *__restrict__ output, const double cut_left,
                          const double cut_right, const int n_slices,
                          const int n_macroparticles)
{
    // Number of Iterations of the inner loop
    const int STEP = 16;
    const double inv_bin_width = n_slices / (cut_right - cut_left);

    // allocate memory for the thread_private histogram
    int **histo = (int **) malloc(omp_get_max_threads() * sizeof(int *));
    histo[0] = (int *) malloc (omp_get_max_threads() * n_slices * sizeof(int));
    for (int i = 0; i < omp_get_max_threads(); i++)
        histo[i] = (*histo + n_slices * i);

    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        memset(histo[id], 0, n_slices * sizeof(int));
        float fbin[STEP];
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            // First calculate the index to update
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = floor((input[i + j] - cut_left) * inv_bin_width);
            }
            // Then update the corresponding bins
            for (int j = 0; j < loop_count; j++) {
                const int bin  = (int) fbin[j];
                if (bin < 0 || bin >= n_slices) continue;
                histo[id][bin] += 1;
            }
        }

        // Reduce to a single histogram
        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t][i];
        }
    }

    // free memory
    free(histo[0]);
    free(histo);
}

extern "C" void histogram_v3(const double *__restrict__ input,
                          double *__restrict__ output, const double cut_left,
                          const double cut_right, const int n_slices,
                          const int n_macroparticles)
{
    // Number of Iterations of the inner loop
    const int STEP = 16;
    const double inv_bin_width = n_slices / (cut_right - cut_left);

    // allocate memory for the thread_private histogram
    int *histo = (int *) calloc(omp_get_max_threads() * n_slices,  sizeof(int));

    #pragma omp parallel
    {
        const int start_i = omp_get_thread_num() * n_slices;
        const int threads = omp_get_num_threads();
        float fbin[STEP];
        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            // First calculate the index to update
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = floor((input[i + j] - cut_left) * inv_bin_width);
            }
            // Then update the corresponding bins
            for (int j = 0; j < loop_count; j++) {
                const int bin  = (int) fbin[j];
                if (bin < 0 || bin >= n_slices) continue;
                histo[start_i + bin] += 1;
            }
        }

        // Reduce to a single histogram
        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t*n_slices + i];
        }
    }

    // free memory
    free(histo);
}

extern "C" void histogram_v4(const double *__restrict__ input,
                          double *__restrict__ output, const double cut_left,
                          const double cut_right, const int n_slices,
                          const int n_macroparticles)
{
    // Number of Iterations of the inner loop
    const int STEP = 16;
    const double inv_bin_width = n_slices / (cut_right - cut_left);
    int *histo;

    // allocate memory for the thread_private histogram

    #pragma omp parallel
    {
        const int start_i = omp_get_thread_num() * n_slices;
        const int threads = omp_get_num_threads();
        #pragma omp master
        histo = (int *) calloc(threads * n_slices,  sizeof(int));
        #pragma omp barrier

        float fbin[STEP];

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i += STEP) {

            const int loop_count = n_macroparticles - i > STEP ?
                                   STEP : n_macroparticles - i;

            // First calculate the index to update
            for (int j = 0; j < loop_count; j++) {
                fbin[j] = floor((input[i + j] - cut_left) * inv_bin_width);
            }
            // Then update the corresponding bins
            for (int j = 0; j < loop_count; j++) {
                const int bin  = (int) fbin[j];
                if (bin < 0 || bin >= n_slices) continue;
                histo[start_i + bin] += 1;
            }
        }

        // Reduce to a single histogram
        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t*n_slices + i];
        }
    }

    // free memory
    free(histo);
}

extern "C" void histogram_v5(const double *__restrict__ input,
                          double *__restrict__ output, const double cut_left,
                          const double cut_right, const int n_slices,
                          const int n_macroparticles)
{
    // Number of Iterations of the inner loop
    const double inv_bin_width = n_slices / (cut_right - cut_left);
    int *histo;

    // allocate memory for the thread_private histogram

    #pragma omp parallel
    {
        const int start_i = omp_get_thread_num() * n_slices;
        const int threads = omp_get_num_threads();
        #pragma omp master
        histo = (int *) calloc(threads * n_slices,  sizeof(int));
        #pragma omp barrier

        #pragma omp for
        for (int i = 0; i < n_macroparticles; i++) {
            int bin =  (int) floor((input[i] - cut_left) * inv_bin_width);
            // Then update the corresponding bins
            if (bin >= 0 && bin < n_slices)
                histo[start_i + bin] += 1;
        }

        // Reduce to a single histogram
        #pragma omp for
        for (int i = 0; i < n_slices; i++) {
            output[i] = 0.;
            for (int t = 0; t < threads; t++)
                output[i] += histo[t*n_slices + i];
        }
    }

    // free memory
    free(histo);
}


extern "C" void drift(double * __restrict__ beam_dt,
                      const double * __restrict__ beam_dE,
                      const double T0, const double length_ratio,
                      const double alpha_zero, const double alpha_one,
                      const double alpha_two,
                      const double beta, const double energy,
                      const int n_macroparticles) {

    int i;
    double T = T0 * length_ratio;

    // if ( strcmp (solver, "simple") == 0 )
    // {
    //     double coeff = eta_zero / (beta * beta * energy);
    //     #pragma omp parallel for
    //     for (int i = 0; i < n_macroparticles; i++)
    //         beam_dt[i] += T * coeff * beam_dE[i];
    // }

    // else if ( strcmp (solver, "legacy") == 0 )
    // {
    //     const double coeff = 1. / (beta * beta * energy);
    //     const double eta0 = eta_zero * coeff;
    //     const double eta1 = eta_one * coeff * coeff;
    //     const double eta2 = eta_two * coeff * coeff * coeff;

    //     if (alpha_order == 0)
    //         for ( i = 0; i < n_macroparticles; i++ )
    //             beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) - 1.);
    //     else if (alpha_order == 1)
    //         for ( i = 0; i < n_macroparticles; i++ )
    //             beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
    //                                      - eta1 * beam_dE[i] * beam_dE[i]) - 1.);
    //     else
    //         for ( i = 0; i < n_macroparticles; i++ )
    //             beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
    //                                      - eta1 * beam_dE[i] * beam_dE[i]
    //                                      - eta2 * beam_dE[i] * beam_dE[i] * beam_dE[i]) - 1.);
    // }

    // else
    // {

        const double invbetasq = 1 / (beta * beta);
        const double invenesq = 1 / (energy * energy);
        // double beam_delta;

        #pragma omp parallel for
        for ( i = 0; i < n_macroparticles; i++ )
        {

            double beam_delta = sqrt(1. + invbetasq *
                              (beam_dE[i] * beam_dE[i] * invenesq + 2.*beam_dE[i] / energy)) - 1.;

            beam_dt[i] += T * (
                              (1. + alpha_zero * beam_delta +
                               alpha_one * (beam_delta * beam_delta) +
                               alpha_two * (beam_delta * beam_delta * beam_delta)) *
                              (1. + beam_dE[i] / energy) / (1. + beam_delta) - 1.);

        }

    // }

}

