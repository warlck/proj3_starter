#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>








// Helper functions -----------------------------------------------------------

/*
 * Get a current timestamp with us accuracy. This will give you the time that
 * has passed since a certain point in time. While the value itself doesn't
 * tell you much, you can subtract timestamps from each other to get the
 * amount of time that has passed between them.
 */

static inline uint64_t timestamp_us() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return 1000000L * tv.tv_sec + tv.tv_usec;
}

// Vol ------------------------------------------------------------------------

// Volumes are used to represent the activations (i.e., state) between the
// different layers of the CNN. They all have three dimensions. The inter-
// pretation of their content depends on the layer that produced them. Before
// the first iteration, the Volume holds the data of the image we want to
// classify (the depth are the three color dimensions). After the last stage
// of the CNN, the Volume holds the probabilities that an image is part of
// a specific category.

/*
 * Represents a three-dimensional array of numbers, and its size. The numbers
 * at (x,y,d) are stored in array w at location ((v->sx * y)+x)*v->depth+d.
 */

typedef struct vol {
  uint64_t sx,sy,depth;
  double* w;
} vol_t;

/*
 * Get the value at a specific entry of the array.
 */

static inline double get_vol(vol_t* v, int x, int y, int d) {
  return v->w[((v->sx * y)+x)*v->depth+d];
}

/*
 * Set the value at a specific entry of the array.
 */

static inline void set_vol(vol_t* v, int x, int y, int d, double val) {
  v->w[((v->sx * y)+x)*v->depth+d] = val;
}

/*
 * Allocate a new array with specific dimensions and default value v.
 */

static vol_t* make_vol(int sx, int sy, int d, double v) {
  vol_t* out = (vol_t*)malloc(sizeof(struct vol));
  out->w = (double*)malloc(sizeof(double)*(sx*sy*d));
  out->sx = sx;
  out->sy = sy;
  out->depth = d;

  for (int x = 0; x < sx; x++)
    for (int y = 0; y < sy; y++)
      for (int z = 0; z < d; z++)
        set_vol(out, x, y, z, v);

  return out;
}

/*
 * Copy the contents of one Volume to another (assuming same dimensions).
 */

static vol_t* copy_vol(vol_t* dest, vol_t* src) {
  for (int x = 0; x < dest->sx; x++)
    for (int y = 0; y < dest->sy; y++)
      for (int z = 0; z < dest->depth; z++)
        set_vol(dest, x, y, z, get_vol(src, x, y, z));
}

/*
 * Deallocate the array.
 */
void free_vol(vol_t* v) {
  free(v->w);
  free(v);
}

// A note about layers --------------------------------------------------------

/*
 * What follows are the different layers of the CNN. You will not have to
 * understand what these layers are actually doing. In general terms, each
 * layer performs a "forward" operation on a batch of inputs. During this
 * forward operation, the layer takes a set of input Volumes and transforms
 * them into a set of output Volumes (one output for each input). What differs
 * is the operation performed by each layer.
 *
 * In addition to the _forward function, each layer also provides a data
 * structure, holding (fixed) parameters for that layer, a make_ function to
 * allocate an instance of the layer with a particular set of parameters and
 * a load function to load training data for that layer from a file. Note that
 * you will not have to make any changes to any of these functions. The only
 * function you need to consider is the _forward function.
 */

// Convolutional Layer --------------------------------------------------------

typedef struct conv_layer {
  // required
  int out_depth;
  int sx;
  int in_depth;
  int in_sx;
  int in_sy;

  // optional
  int sy;
  int stride;
  int pad;
  double l1_decay_mul;
  double l2_decay_mul;

  // computed
  int out_sx;
  int out_sy;
  double bias;
  vol_t* biases;
  vol_t** filters;
} conv_layer_t;

conv_layer_t* make_conv_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int filters, int stride, int pad) {
  conv_layer_t* l = (conv_layer_t*)malloc(sizeof(conv_layer_t));

  // required
  l->out_depth = filters;
  l->sx = sx;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;
    
  // optional
  l->sy = l->sx;
  l->stride = stride;
  l->pad = pad;
  l->l1_decay_mul = 0.0;
  l->l2_decay_mul = 1.0;

  // computed
  l->out_sx = floor((l->in_sx + l->pad * 2 - l->sx) / l->stride + 1);
  l->out_sy = floor((l->in_sy + l->pad * 2 - l->sy) / l->stride + 1);

  l->filters = (vol_t**)malloc(sizeof(vol_t*)*filters);
  for (int i = 0; i < filters; i++) {
    l->filters[i] = make_vol(l->sx, l->sy, l->in_depth, 0.0);
  }

  l->bias = 0.0;
  l->biases = make_vol(1, 1, l->out_depth, l->bias);

  return l;
}


void conv_forward_dloop(conv_layer_t* l, double *Vwpp, int V_sx, int V_sy, int Vdepth, 
 int xy_stride, vol_t* A, int d) 
{
    vol_t* f = l->filters[d];
    int fdepth = f -> depth;
    int fsx = f -> sx;
    int fsy = f -> sy;

    // printf("d = %d, fyx = %d\n",d, fdepth);
    double *fwpp = f->w;

    int x = -l->pad;
    int y = -l->pad;

    double wd = l->biases->w[d];

    for(int ay = 0; ay < l->out_sy; y += xy_stride, ay++) {

      x = -l->pad;

      for(int ax=0; ax < l->out_sx; x += xy_stride, ax++) {

        double a = 0.0;

        for(int fy = 0; fy < fsy; fy++) {

          int oy = y + fy;

          // Preliminary calculation of fw and Vw pointers

          double *fwp = fwpp + (fsx * fy)*fdepth;
          double *Vwp = Vwpp + (V_sx * oy)*Vdepth;

          for(int fx = 0; fx < fsx; fx++) {

            int ox = x + fx;

            if(oy >= 0 && oy < V_sy && ox >=0 && ox < V_sx) {

              // double *fw = f->w + (f->sx * fy)*f->depth;
              // double *Vw = V->w + (V_sx * oy)*V->depth ;

              // fw +=  fx*f->depth;
              // Vw +=  ox*V->depth;

              double * fw = fwp + fx*fdepth;
              double * Vw = Vwp + ox*Vdepth;

              

              for(int fd=0;fd < fdepth; fd++) {

                // a += f->w[((f->sx * fy)+fx)*f->depth+fd] * V->w[((V_sx * oy)+ox)*V->depth+fd];
                a += fw[fd]*Vw[fd];

              }
            }
          }
        }

        a += wd;
        set_vol(A, ax, ay, d, a);

      }
    }
}


void conv_forward_dloop_vectorized(conv_layer_t* l, __m128i Vwpp_vector, int V_sx,  __m128i V_sx_vector,
               int V_sy, __m128i V_sy_vector,  __m128i V_depth_vector,  int xy_stride, __m128i A_vector, int d) 
{
    vol_t* f = l->filters[d];
    // vol_t** lfilters = l->filters;

    int fdepth = f -> depth;
    int fsx = f -> sx;
    int fsy = f -> sy;



    double *fwpp = f->w;
    int x = -l->pad;
    int y = -l->pad;

    __m128d wd_vector0 =  _mm_load1_pd(l->biases->w + d);
  


    for(int ay = 0; ay < l->out_sy; y += xy_stride, ay++) {

      x = -l->pad;

      for(int ax=0; ax < l->out_sx; x += xy_stride, ax++) {

        __m128d a0 = _mm_setzero_pd();
        __m128d a1 = _mm_setzero_pd();

        for(int fy = 0; fy < fsy; fy++) {

          int oy = y + fy;

          // Preliminary calculation of fw and Vw pointers

          double *fwp = fwpp + (fsx * fy)*fdepth;


          // double *Vwp = Vwpp + (V_sx * oy)*Vdepth;
          __m128i oy_vector = _mm_set1_epi32(oy);
          __m128i product = _mm_mullo_epi32(V_sx_vector, oy_vector);
          
          product = _mm_mullo_epi32(product, V_depth_vector);
          __m128i Vwp_vector = _mm_add_epi32(Vwpp_vector, product);




          for(int fx = 0; fx < fsx; fx++) {

            int ox = x + fx;

            if(oy >= 0 && oy < V_sy && ox >=0 && ox < V_sx) {

              // double *fw = f->w + (f->sx * fy)*f->depth;
              // double *Vw = V->w + (V_sx * oy)*V->depth ;

              // fw +=  fx*f->depth;
              // Vw +=  ox*V->depth;

              __m128i ox_vector = _mm_set1_epi32(ox); 


              double * fw = fwp + fx*fdepth;

              __m128i vwproduct =  _mm_mullo_epi32(ox_vector, V_depth_vector);
              __m128i Vw_vector = _mm_add_epi32(Vwp_vector, vwproduct);

              // double * Vw = Vwp + ox*Vdepth;

              

              for(int fd=0;fd < fdepth; fd++) {

                // a += f->w[((f->sx * fy)+fx)*f->depth+fd] * V->w[((V_sx * oy)+ox)*V->depth+fd];
                __m128d fwd0 = _mm_load1_pd(fw + fd);
                __m128d fwd1 = _mm_load1_pd(fw + fd);

                double Vwd[] = {((double *)(Vw_vector[0]))[fd], ((double *)(Vw_vector[1]))[fd], ((double *)(Vw_vector[2]))[fd], ((double *)(Vw_vector[3]))[fd]};
                __m128d Vwd0 = _mm_loadu_pd(Vwd);
                __m128d Vwd1 = _mm_loadu_pd(Vwd + 2);

                a0  = _mm_add_pd(a0,_mm_mul_pd(Vwd0, fwd0));
                a1  = _mm_add_pd(a1,_mm_mul_pd(Vwd1, fwd1));

                // a += fw[fd]*Vw[fd];

              }
            }
          }
        }

        a0 = _mm_add_pd(a0, wd_vector0);
        a1 = _mm_add_pd(a1, wd_vector0);
        // a += wd;

        for (int i = 0; i < 2; i++) {
          // loop chooses 0th and 2nd element of A_vector in first loop and 1st and 3rd elements 
          // in second loop  due to our choice of 'a' values that need to be stored in respective
          // volumes. 


          vol_t * A0 = A_vector[i]; 
          // double a0_val = ((double *)a0)[i];
          double a0_val = a0[i];
          set_vol(A0, ax, ay, d, a0_val);

          vol_t * A1 = A_vector[i+2];
          double a1_val = a1[i];
          set_vol(A0, ax, ay, d, a1_val);

        }

        // set_vol(A, ax, ay, d, a);

      }
    }
}

void conv_forward_iloop(conv_layer_t* l, vol_t** in, vol_t** out, int i) {
  vol_t* V = in[i];

  vol_t* A = out[i];
      
  int V_sx = V->sx;
  int V_sy = V->sy;
  int Vdepth = V -> depth;

  // printf("i = %d, V_sx = %d\n",i, V_sx);

  int xy_stride = l->stride;

  double *Vwpp = V->w;



  
  int l_out_depth = l->out_depth;

  for(int d = 0; d < l_out_depth/4*4; d+=4) {
      conv_forward_dloop(l, Vwpp, V_sx, V_sy, Vdepth, xy_stride, A, d + 0);
      conv_forward_dloop(l, Vwpp, V_sx, V_sy, Vdepth, xy_stride, A, d + 1);
      conv_forward_dloop(l, Vwpp, V_sx, V_sy, Vdepth, xy_stride, A, d + 2);
      conv_forward_dloop(l, Vwpp, V_sx, V_sy, Vdepth, xy_stride, A, d + 3);
  }

    for(int d = l_out_depth/4*4; d < l_out_depth; d++) {
      conv_forward_dloop(l, Vwpp, V_sx, V_sy, Vdepth, xy_stride, A, d);
   }
}

void conv_forward_iloop_vectorized(conv_layer_t* l, vol_t** in, vol_t** out, int i) {
  vol_t* V = in[i];
  __m128i V0 =  _mm_loadu_si128( (__m128i *) (in + i));

    vol_t* A = out[i];
  __m128i A_vector =_mm_loadu_si128( (__m128i *) (out + i));
 
  
  int V_sx = V->sx;
  int V_sy = V->sy;
  int Vdepth = V -> depth;
    
  __m128i V_sx_vector =  _mm_set1_epi32(V_sx);
  __m128i V_sy_vector =  _mm_set1_epi32(V_sy);
  __m128i V_depth_vector = _mm_set1_epi32(Vdepth);

  
  int xy_stride = l->stride;

  double *Vwpp = V->w;
  double *Vwpp_temp[] = {in[i]->w, in[i+1]->w, in[i+2]->w, in[i+3]->w};

  __m128i Vwpp_vector =  _mm_loadu_si128((__m128i *) Vwpp_temp);

  // int Vwpp_diff = &(V->w) - (uint64_t *)V;

  // __m128i Vwpp_vector = _mm_add_epi32(V0, _mm_set1_epi32(Vwpp_diff));

  
  int l_out_depth = l->out_depth;

  for(int d = 0; d < l_out_depth/4*4; d+=4) {
      conv_forward_dloop_vectorized(l, Vwpp_vector, V_sx,V_sx_vector,  
        V_sy, V_sy_vector, V_depth_vector, xy_stride, A_vector, d + 0);
      // conv_forward_dloop(l, Vwpp, V_sx, V_sy, Vdepth, xy_stride, A, d + 1);
      // conv_forward_dloop(l, Vwpp, V_sx, V_sy, Vdepth, xy_stride, A, d + 2);
      // conv_forward_dloop(l, Vwpp, V_sx, V_sy, Vdepth, xy_stride, A, d + 3);
  }

    for(int d = l_out_depth/4*4; d < l_out_depth; d++) {
      conv_forward_dloop(l, Vwpp, V_sx, V_sy, Vdepth, xy_stride, A, d);
   }
}

void conv_forward(conv_layer_t* l, vol_t** in, vol_t** out, int start, int end) {


  // LOOP iterating with "i"

  for (int i = start; i < end/4*4; i += 4) {

    conv_forward_iloop(l, in, out, i + 0);
    conv_forward_iloop(l, in, out, i + 1);
    conv_forward_iloop(l, in, out, i + 2);
    conv_forward_iloop(l, in, out, i + 3);

    // vol_t* V = in[i];
    // vol_t* A = out[i];
        
    // int V_sx = V->sx;
    // int V_sy = V->sy;
    // int Vdepth = V -> depth;

    // int xy_stride = l->stride;

    // double *Vwpp = V->w;
    
    // for(int d = 0; d < l->out_depth; d++) {

    //   vol_t* f = l->filters[d];
    //   int fdepth = f -> depth;
    //   int fsx = f -> sx;
    //   int fsy = f -> sy;

    //   double *fwpp = f->w;

    //   int x = -l->pad;
    //   int y = -l->pad;

    //   double wd = l->biases->w[d];

    //   for(int ay = 0; ay < l->out_sy; y += xy_stride, ay++) {

    //     x = -l->pad;

    //     for(int ax=0; ax < l->out_sx; x += xy_stride, ax++) {

    //       double a = 0.0;

    //       for(int fy = 0; fy < fsy; fy++) {

    //         int oy = y + fy;

    //         // Preliminary calculation of fw and Vw pointers

    //         double *fwp = fwpp + (fsx * fy)*fdepth;
    //         double *Vwp = Vwpp + (V_sx * oy)*Vdepth;

    //         for(int fx = 0; fx < fsx; fx++) {

    //           int ox = x + fx;

    //           if(oy >= 0 && oy < V_sy && ox >=0 && ox < V_sx) {

    //             // double *fw = f->w + (f->sx * fy)*f->depth;
    //             // double *Vw = V->w + (V_sx * oy)*V->depth ;

    //             // fw +=  fx*f->depth;
    //             // Vw +=  ox*V->depth;

    //             double * fw = fwp + fx*fdepth;
    //             double * Vw = Vwp + ox*Vdepth;

                

    //             for(int fd=0;fd < fdepth; fd++) {

    //               // a += f->w[((f->sx * fy)+fx)*f->depth+fd] * V->w[((V_sx * oy)+ox)*V->depth+fd];
    //               a += fw[fd]*Vw[fd];

    //             }
    //           }
    //         }
    //       }

    //       a += wd;
    //       set_vol(A, ax, ay, d, a);

    //     }
    //   }
    // }
  } 

  for (int i = end/4*4; i <= end; i++ ) {
    conv_forward_iloop(l, in, out, i);
  }
} 


void conv_load(conv_layer_t* l, const char* fn) {
  int sx, sy, depth, filters;

  FILE* fin = fopen(fn, "r");

  fscanf(fin, "%d %d %d %d", &sx, &sy, &depth, &filters);
  assert(sx == l->sx);
  assert(sy == l->sy);
  assert(depth == l->in_depth);
  assert(filters == l->out_depth);

  for(int d = 0; d < l->out_depth; d++)
    for (int x = 0; x < sx; x++)
      for (int y = 0; y < sy; y++)
        for (int z = 0; z < depth; z++) {
          double val;
          fscanf(fin, "%lf", &val);
          set_vol(l->filters[d], x, y, z, val);
        }

  for(int d = 0; d < l->out_depth; d++) {
    double val;
    fscanf(fin, "%lf", &val);
    set_vol(l->biases, 0, 0, d, val);
  }

  fclose(fin);
}

// Relu Layer -----------------------------------------------------------------

typedef struct relu_layer {
  // required
  int in_depth;
  int in_sx;
  int in_sy;

  // computed
  int out_depth;
  int out_sx;
  int out_sy;
} relu_layer_t;

relu_layer_t* make_relu_layer(int in_sx, int in_sy, int in_depth) {
  relu_layer_t* l = (relu_layer_t*)malloc(sizeof(relu_layer_t));

  // required
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // computed
  l->out_sx = l->in_sx;
  l->out_sy = l->in_sy;
  l->out_depth = l->in_depth;

  return l;
}

void relu_forward(relu_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int j = start; j <= end; j++) {
    for (int i = 0; i < l->in_sx*l->in_sy*l->in_depth; i++) {
      out[j]->w[i] = (in[j]->w[i] < 0.0) ? 0.0 : in[j]->w[i];
    }
  }
}

// Pool Layer -----------------------------------------------------------------

typedef struct pool_layer {
  // required
  int sx;
  int in_depth;
  int in_sx;
  int in_sy;

  // optional
  int sy;
  int stride;
  int pad;

  // computed
  int out_depth;
  int out_sx;
  int out_sy;
} pool_layer_t;

pool_layer_t* make_pool_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int stride) {
  pool_layer_t* l = (pool_layer_t*)malloc(sizeof(pool_layer_t));

  // required
  l->sx = sx;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // optional
  l->sy = l->sx;
  l->stride = stride;
  l->pad = 0;

  // computed
  l->out_depth = in_depth;
  l->out_sx = floor((l->in_sx + l->pad * 2 - l->sx) / l->stride + 1);
  l->out_sy = floor((l->in_sy + l->pad * 2 - l->sy) / l->stride + 1);

  return l;
}

void pool_forward(pool_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];
        
    int n=0;
    for(int d=0;d<l->out_depth;d++) {
      int x = -l->pad;
      int y = -l->pad;
      for(int ax=0; ax<l->out_sx; x+=l->stride,ax++) {
        y = -l->pad;
        for(int ay=0; ay<l->out_sy; y+=l->stride,ay++) {
  
          double a = -99999;
          for(int fx=0;fx<l->sx;fx++) {
            for(int fy=0;fy<l->sy;fy++) {
              int oy = y+fy;
              int ox = x+fx;
              if(oy>=0 && oy<V->sy && ox>=0 && ox<V->sx) {
                double v = get_vol(V, ox, oy, d);
                if(v > a) { a = v; }
              }
            }
          }
          n++;
          set_vol(A, ax, ay, d, a);
        }
      }
    }
  }
}

// FC Layer -------------------------------------------------------------------

typedef struct fc_layer {
  // required
  int out_depth;
  int in_depth;
  int in_sx;
  int in_sy;

  // optional
  double l1_decay_mul;
  double l2_decay_mul;

  // computed
  int out_sx;
  int out_sy;
  int num_inputs;
  double bias;
  vol_t* biases;
  vol_t** filters;
} fc_layer_t;

fc_layer_t* make_fc_layer(int in_sx, int in_sy, int in_depth,
                          int num_neurons) {
  fc_layer_t* l = (fc_layer_t*)malloc(sizeof(fc_layer_t));

  // required
  l->out_depth = num_neurons;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;
    
  // optional
  l->l1_decay_mul = 0.0;
  l->l2_decay_mul = 1.0;

  // computed
  l->num_inputs = l->in_sx * l->in_sy * l->in_depth;
  l->out_sx = 1;
  l->out_sy = 1;

  l->filters = (vol_t**)malloc(sizeof(vol_t*)*num_neurons);
  for (int i = 0; i < l->out_depth; i++) {
    l->filters[i] = make_vol(1, 1, l->num_inputs, 0.0);
  }

  l->bias = 0.0;
  l->biases = make_vol(1, 1, l->out_depth, l->bias);

  return l;
}

void fc_forward(fc_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int j = start; j <= end; j++) {
    vol_t* V = in[j];
    vol_t* A = out[j];
        
    for(int i=0;i<l->out_depth;i++) {
      double a = 0.0;
      for(int d=0;d<l->num_inputs;d++) {
        a += V->w[d] * l->filters[i]->w[d];
      }
      a += l->biases->w[i];
      A->w[i] = a;
    }
  }
}

void fc_load(fc_layer_t* l, const char* fn) {
  FILE* fin = fopen(fn, "r");

  int num_inputs;
  int out_depth;
  fscanf(fin, "%d %d", &num_inputs, &out_depth);
  assert(out_depth == l->out_depth);
  assert(num_inputs == l->num_inputs);

  for(int i = 0; i < l->out_depth; i++)
    for(int d = 0; d < l->num_inputs; d++) {
      double val;
      fscanf(fin, "%lf", &val);
      l->filters[i]->w[d] = val;
    }

  for(int i = 0; i < l->out_depth; i++) {
    double val;
    fscanf(fin, "%lf", &val);
    l->biases->w[i] = val;
  }

  fclose(fin);
}

// Softmax Layer --------------------------------------------------------------

// Maximum supported out_depth
#define MAX_ES 16

typedef struct softmax_layer {
  // required
  int in_depth;
  int in_sx;
  int in_sy;
  double* es; 

  // computed
  int out_depth;
  int out_sx;
  int out_sy;
} softmax_layer_t;

softmax_layer_t* make_softmax_layer(int in_sx, int in_sy, int in_depth) {
  softmax_layer_t* l = (softmax_layer_t*)malloc(sizeof(softmax_layer_t));

  // required
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // computed
  l->out_sx = 1;
  l->out_sy = 1;
  l->out_depth = l->in_sx * l->in_sy * l->in_depth;

  l->es = (double*)malloc(sizeof(double)*l->out_depth);

  return l;
}

void softmax_forward(softmax_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  double es[MAX_ES];

  for (int j = start; j <= end; j++) {
    vol_t* V = in[j];
    vol_t* A = out[j];
  
    // compute max activation
    double amax = V->w[0];
    for(int i=1;i<l->out_depth;i++) {
      if(V->w[i] > amax) amax = V->w[i];
    }
  
    // compute exponentials (carefully to not blow up)
    double esum = 0.0;
    for(int i=0;i<l->out_depth;i++) {
      double e = exp(V->w[i] - amax);
      esum += e;
      es[i] = e;
    }
  
    // normalize and output to sum to one
    for(int i=0;i<l->out_depth;i++) {
      es[i] /= esum;
      A->w[i] = es[i];
    }
  }
}

// Neural Network -------------------------------------------------------------

/*
 * This represents the CNN we will use in this project. It consists of 11
 * layers, which means that there are 12 volumes of data (where the first one
 * is the input data and the last one the classification result).
 */

#define LAYERS 11

typedef struct network {
  vol_t* v[LAYERS+1];
  conv_layer_t* l0;
  relu_layer_t* l1;
  pool_layer_t* l2;
  conv_layer_t* l3;
  relu_layer_t* l4;
  pool_layer_t* l5;
  conv_layer_t* l6;
  relu_layer_t* l7;
  pool_layer_t* l8;
  fc_layer_t* l9;
  softmax_layer_t* l10;
} network_t;

/*
 * Instantiate our specific CNN.
 */

network_t* make_network() {
  network_t* net = (network_t*)malloc(sizeof(network_t));
  net->v[0] = make_vol(32, 32, 3, 0.0);
  net->l0 = make_conv_layer(32, 32, 3, 5, 16, 1, 2);
  net->v[1] = make_vol(net->l0->out_sx, net->l0->out_sy, net->l0->out_depth, 0.0);
  net->l1 = make_relu_layer(net->v[1]->sx, net->v[1]->sy, net->v[1]->depth);
  net->v[2] = make_vol(net->l1->out_sx, net->l1->out_sy, net->l1->out_depth, 0.0);
  net->l2 = make_pool_layer(net->v[2]->sx, net->v[2]->sy, net->v[2]->depth, 2, 2);
  net->v[3] = make_vol(net->l2->out_sx, net->l2->out_sy, net->l2->out_depth, 0.0);
  net->l3 = make_conv_layer(net->v[3]->sx, net->v[3]->sy, net->v[3]->depth, 5, 20, 1, 2);
  net->v[4] = make_vol(net->l3->out_sx, net->l3->out_sy, net->l3->out_depth, 0.0);
  net->l4 = make_relu_layer(net->v[4]->sx, net->v[4]->sy, net->v[4]->depth);
  net->v[5] = make_vol(net->l4->out_sx, net->l4->out_sy, net->l4->out_depth, 0.0);
  net->l5 = make_pool_layer(net->v[5]->sx, net->v[5]->sy, net->v[5]->depth, 2, 2);
  net->v[6] = make_vol(net->l5->out_sx, net->l5->out_sy, net->l5->out_depth, 0.0);
  net->l6 = make_conv_layer(net->v[6]->sx, net->v[6]->sy, net->v[6]->depth, 5, 20, 1, 2);
  net->v[7] = make_vol(net->l6->out_sx, net->l6->out_sy, net->l6->out_depth, 0.0);
  net->l7 = make_relu_layer(net->v[7]->sx, net->v[7]->sy, net->v[7]->depth);
  net->v[8] = make_vol(net->l7->out_sx, net->l7->out_sy, net->l7->out_depth, 0.0);
  net->l8 = make_pool_layer(net->v[8]->sx, net->v[8]->sy, net->v[8]->depth, 2, 2);
  net->v[9] = make_vol(net->l8->out_sx, net->l8->out_sy, net->l8->out_depth, 0.0);
  net->l9 = make_fc_layer(net->v[9]->sx, net->v[9]->sy, net->v[9]->depth, 10);
  net->v[10] = make_vol(net->l9->out_sx, net->l9->out_sy, net->l9->out_depth, 0.0);
  net->l10 = make_softmax_layer(net->v[10]->sx, net->v[10]->sy, net->v[10]->depth);
  net->v[11] = make_vol(net->l10->out_sx, net->l10->out_sy, net->l10->out_depth, 0.0);
  return net;
}

/* 
 * Free our specific CNN.
 */

void free_network(network_t* net) {
  for (int i = 0; i < LAYERS+1; i++)
    free_vol(net->v[i]);

  free(net->l0);
  free(net->l1);
  free(net->l2);
  free(net->l3);
  free(net->l4);
  free(net->l5);
  free(net->l6);
  free(net->l7);
  free(net->l8);
  free(net->l9);
  free(net->l10);

  free(net);
}

/*
 * We organize data as "batches" of volumes. Each batch consists of a number of samples,
 * each of which contains a volume for every intermediate layer. Say we have L layers
 * and a set of N input images. Then batch[l][n] contains the volume at layer l for
 * input image n.2
 *
 * By using batches, we can process multiple images at once in each run of the forward
 * functions of the different layers.
 */

typedef vol_t** batch_t;

/*
 * This function allocates a new batch for the network old_net with size images.
 */

batch_t* make_batch(network_t* old_net, int size) {
  batch_t* out = (batch_t*)malloc(sizeof(vol_t**)*(LAYERS+1));
  for (int i = 0; i < LAYERS+1; i++) {
    out[i] = (vol_t**)malloc(sizeof(vol_t*)*size);
    for (int j = 0; j < size; j++) {
      out[i][j] = make_vol(old_net->v[i]->sx, old_net->v[i]->sy, old_net->v[i]->depth, 0.0);
    }
  }

  return out;
}

/*
 * Free a previously allocated batch.
 */

void free_batch(batch_t* v, int size) {
  for (int i = 0; i < LAYERS+1; i++) {
    for (int j = 0; j < size; j++) {
      free_vol(v[i][j]);
    }
    free(v[i]);
  }
  free(v);
}

/*
 * Apply our network to a specific batch of inputs. The batch has to be given
 * as input to v and start/end are the first and the last image in that batch
 * to process (start and end are inclusive).
 */

static   double l0_time = 0;
static   double l1_time = 0;
static   double l2_time = 0;
static   double l3_time = 0;
static   double l4_time = 0;
static   double l5_time = 0;
static   double l6_time = 0;
static   double l7_time = 0;
static   double l8_time = 0;
static   double l9_time = 0;
static   double l10_time = 0;
static   double total_execution_time  = 0;

void net_forward(network_t* net, batch_t* v, int start, int end) {
  // uint64_t t0 = timestamp_us();
  conv_forward(net->l0, v[0], v[1], start, end);
  // uint64_t t1 = timestamp_us();
  relu_forward(net->l1, v[1], v[2], start, end);
  // uint64_t t2 = timestamp_us();
  pool_forward(net->l2, v[2], v[3], start, end);
  // uint64_t t3 = timestamp_us();
  conv_forward(net->l3, v[3], v[4], start, end);
  // uint64_t t4 = timestamp_us();
  relu_forward(net->l4, v[4], v[5], start, end);
  // uint64_t t5 = timestamp_us();
  pool_forward(net->l5, v[5], v[6], start, end);
  // uint64_t t6 = timestamp_us();
  conv_forward(net->l6, v[6], v[7], start, end);
  // uint64_t t7 = timestamp_us();
  relu_forward(net->l7, v[7], v[8], start, end);
  // uint64_t t8 = timestamp_us();
  pool_forward(net->l8, v[8], v[9], start, end);
  // uint64_t t9 = timestamp_us();
  fc_forward(net->l9, v[9], v[10], start, end);
  // uint64_t t10 = timestamp_us();
  softmax_forward(net->l10, v[10], v[11], start, end);
  // uint64_t t11 = timestamp_us();


  // l0_time += t1 - t0;
  // l1_time += t2 - t1;
  // l2_time += t3 - t2;
  // l3_time += t4 - t3;
  // l4_time += t5 - t4;
  // l5_time += t6 - t5;
  // l6_time += t7 - t6;
  // l7_time += t8 - t7;
  // l8_time += t9 - t8;
  // l9_time += t10 - t9;
  // l10_time += t11 - t10;
  // total_execution_time += t11 - t0;

}

/*
 * Putting everything together: Take a set of n input images as 3-dimensional
 * Volumes and process them using the CNN in batches of 1. Then look at the
 * output (which is a set of 10 labels, each of which tells us the likelihood
 * of a specific category) and classify the image as a cat iff the likelihood
 * of "cat" is larger than 50%. Writes the cat likelihood for all images into
 * an output array (0 = definitely no cat, 1 = definitely cat).
 */

#define CAT_LABEL 3
void net_classify_cats(network_t* net, vol_t** input, double* output, int n) {
  // uint64_t tbb = timestamp_us();
  int batchSize = n;
  batch_t* batch = make_batch(net, batchSize);
  // uint64_t tab  = timestamp_us();
  // printf("batch time %u\n", tab - tbb);

  for (int i = 0; i < n; i++) 
    copy_vol(batch[0][i] , input[i]);

  net_forward(net, batch, 0, batchSize - 1);

  for (int i = 0; i < n; i++) 
    output[i] = batch[11][i]->w[CAT_LABEL]; 
  

  // l0_time /= 1.0*n;
  // l1_time /= 1.0*n;
  // l2_time /= 1.0*n;
  // l3_time /= 1.0*n;
  // l4_time /= 1.0*n;
  // l5_time /= 1.0*n;
  // l6_time /= 1.0*n;
  // l7_time /= 1.0*n;
  // l8_time /= 1.0*n;
  // l9_time /= 1.0*n;;
  // l10_time /= 1.0*n;
  // total_execution_time /= 1.0*n;

  // printf(" average l0 time = %lf\n", l0_time/total_execution_time*100 );
  // printf(" average l1 time = %lf\n", l1_time/total_execution_time*100 );
  // printf(" average l2 time = %lf\n", l2_time/total_execution_time*100 );
  // printf(" average l3 time = %lf\n", l3_time/total_execution_time*100 );
  // printf(" average l4 time = %lf\n", l4_time/total_execution_time*100 );
  // printf(" average l5 time = %lf\n", l5_time/total_execution_time*100 );
  // printf(" average l6 time = %lf\n", l6_time/total_execution_time*100 );
  // printf(" average l7 time = %lf\n", l7_time/total_execution_time*100 );
  // printf(" average l8 time = %lf\n", l8_time/total_execution_time*100 );
  // printf(" average l9 time = %lf\n", l9_time/total_execution_time*100 );
  // printf(" average l10 time = %lf\n", l10_time/total_execution_time*100 );
  // printf("average total execution time = %lf\n", total_execution_time);



  free_batch(batch, 1);
}

// IGNORE EVERYTHING BELOW THIS POINT -----------------------------------------

// Including C files in other C files is very bad style and should be avoided
// in any real application. We do it here since we want everything that you
// may edit to be in one file, without having to fix the interfaces between
// the different components of the system.

#include "util.c"
#include "main.c"
