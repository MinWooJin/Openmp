// openmp.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//
#include "stdafx.h"
#include <stdio.h>
#include <omp.h>
#include <emmintrin.h>
#include <vector>

#define MAT_SIZE	400

using namespace std;

void gen_init(double *a, double *b, int n, int hi)
{
  printf("Generating input ...\n");
  srand(1234);
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      a[i*n + j] = rand() % hi;
      b[i*n + j] = rand() % hi;
    }
  }
}

void mat_mult_default(double *a, double *b, double *c, int n)
{
  printf("Running default matrix multiplication ...\n");
  double t;
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      t = 0.0;
      for (int k=0; k<n; k++) {
        t += a[i*n + k] * b[k*n + j];
	}
      c[i*n + j]=t;
    }
  }
}

void mat_mult_simd(double *a, double *b, double *c, int n)
{
  printf("Running SIMD matrix multiplication ...\n");
  
  int i, j, k;
 
  __m128d c1,b1,a1;
  int ida = 2;

#pragma omp parallel private (i, j, k, a1, b1, c1)
{
#pragma omp for
  for (i=0; i<n; i++) {
	  for (j=0; j<n/2; j++) {
		 	  c1 = _mm_load_pd(c+i*n+j*ida);
		  for (k=0; k<n; k++) {
			  b1 = _mm_load_pd(b+k*n+j*ida);
			  a1 = _mm_load1_pd(a+i*n+k);
			  c1 = _mm_add_pd(c1, _mm_mul_pd(a1,b1));
		  }
		  _mm_store_pd(c+i*n+j*ida,c1);
	  }
  }
}

  int checksize, checksize2, i, j, cloadnum, k, bloadnum, mulnum, csavenum;
  if (MAT_SIZE%2000 == 0) checksize=1000;
  else if (MAT_SIZE%1000 == 0) checksize=500;
  else if (MAT_SIZE%100 == 0) checksize=50;
  else if (MAT_SIZE%50 == 0) checksize=25;
  else if (MAT_SIZE%10 == 0) checksize=5;
  else if (MAT_SIZE%8 == 0) checksize=4;
  else if (MAT_SIZE%4 == 0) checksize=2;
  else if (MAT_SIZE%2 == 0) checksize=1;
  else {
	printf("짝수만 입력\n");
	exit(1);
  }
  checksize2=checksize*2;
 
  __m128d c1[1000],b1[1000],a1;
 
  int ida = 2;

#pragma omp parallel private(i,j,k)
{
#pragma omp for
  for (i=0; i<n; i++) {
	  for (j=0; j<n/checksize2; j++) {
		  for (cloadnum=0; cloadnum<checksize; cloadnum++)
		 	  c1[cloadnum] = _mm_load_pd(c+i*n+(checksize*j+cloadnum)*ida);
		  for (k=0; k<n; k++) {
			  for (bloadnum=0; bloadnum<checksize; bloadnum++)
				b1[bloadnum] = _mm_load_pd(b+k*n+(checksize*j+bloadnum)*ida);
			  a1 = _mm_load1_pd(a+i*n+k);
			  for (mulnum=0; mulnum<checksize; mulnum++)
				c1[mulnum] = _mm_add_pd(c1[mulnum], _mm_mul_pd(a1,b1[mulnum]));
		  }
		  for (csavenum=0; csavenum<checksize; csavenum++)
			  _mm_store_pd(c+i*n+(checksize*j+csavenum)*ida,c1[csavenum]);
	  }
  }
}

}


int main()
{
  vector<double> a(MAT_SIZE*MAT_SIZE);
  vector<double> b(MAT_SIZE*MAT_SIZE);
  vector<double> res_default(MAT_SIZE*MAT_SIZE, 0.0);
  vector<double> res_simd(MAT_SIZE*MAT_SIZE, 0.0);

  gen_init(a.data(), b.data(), MAT_SIZE, 100);

    mat_mult_default(a.data(), b.data(), res_default.data(), MAT_SIZE);

    mat_mult_simd(a.data(), b.data(), res_simd.data(), MAT_SIZE);
    //mat_mult_default(a.data(), b.data(), res_simd.data(), MAT_SIZE);
  
  if (res_default != res_simd) 
    printf("Does not match, failed\n");
  else {
    printf("Success!!\n");
    printf("execution time\n");
  }
  return 0;
}



