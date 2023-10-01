#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <string>
#include <math.h>
#include <algorithm>
#include <random>
#include <SFML/Graphics.hpp>

using namespace std;

//In this work, i means row and j means column

vector <double> infile;
vector <double> x;
vector <double> b;
vector <double> ATb;
vector <double> LU_parameter;
vector <double> coeff;
vector <double> coeff_1;
vector <double> coeff_temp1;
vector <double> coeff_temp2;
vector <double> coeff_temp3;

void read_file();
void matrix_transpose(double **matrix1, double **matrix2, int n);//matrix2 = matrix1_T
void matrix_mul(double **matrix1, double **matrix2, double **matrix3, int n); //matrix1 * matrix2 = matrix3
void matrix_LU(double **matrix1, double **matrix2, double **matrix3, int n); //matrix1 = matrix3 * matrix2
void matrix_copy(double **matrix1, double **matrix2, int n);    //Copy matrix1 to matrix2
void matrix_reset(double **matrix, int n);
void vector_minus(vector <double> &vec1, vector <double> &vec2, vector <double> &vec3, int n); //vec1 - vec2 = vec3
void matrix_mul_vector(double **matrix, vector <double> &vec1, vector <double> &vec2, int n, int size); //matrix*vec1=vec2, matrix is n*size

int main()
{
    //**************************0.For File Input*************************//
    read_file();

    for (int i = 0; i < infile.size(); i++) {
        if(i%2 == 0) x.push_back(infile[i]);
        else b.push_back(infile[i]);
    }
    infile.clear();


    int n_poly, lambda;
    printf("The Number of Polynomial Basis : ");
    scanf("%d", &n_poly);

    printf("Lambda : ");
    scanf("%d", &lambda);

    //**************************1.Dynamic Array*************************//
    //original A
    double **matrix = new double*[x.size()];
    for(int i = 0; i < x.size(); i++)
        matrix[i] = new double[n_poly];

    for(int i=0; i < x.size(); i++)
    {
        for(int j=0; j < n_poly; j++)
        {
            matrix[i][j] = pow(x[i], n_poly-1-j);
        }
    }


    //A_T
    double **matrix_T = new double*[n_poly];
    for(int i = 0; i < n_poly; i++)
        matrix_T[i] = new double[x.size()];

    for(int i=0; i < n_poly; i++)
    {
        for(int j=0; j < x.size(); j++)
        {
            matrix_T[i][j] = matrix[j][i];
        }
    }

    /*
    printf("Print AT\n");
    for(int i=0; i < n_poly; i++)
    {
        for(int j=0; j < x.size(); j++)
        {
            printf("%lf ", matrix_T[i][j]);
        }
        printf("\n");
    }
    */

    //Dynamic Array Temp1
    double **matrix_temp1 = new double*[n_poly];
    for(int i = 0; i < n_poly; i++)
        matrix_temp1[i] = new double[n_poly];
    
    //Dynamic Array Temp2
    double **matrix_temp2 = new double*[n_poly];
    for(int i = 0; i < n_poly; i++)
        matrix_temp2[i] = new double[n_poly];

    //Dynamic Array Temp3
    double **matrix_temp3 = new double*[n_poly];
    for(int i = 0; i < n_poly; i++)
        matrix_temp3[i] = new double[n_poly];

    //Dynamic Array ATA
    double **matrix_ATA = new double*[n_poly];
    for(int i = 0; i < n_poly; i++)
        matrix_ATA[i] = new double[n_poly];

    //Dynamic Array ATA_inv
    double **matrix_ATA_inv = new double*[n_poly];
    for(int i = 0; i < n_poly; i++)
        matrix_ATA_inv[i] = new double[n_poly];

    
    //**************************2.Matrix Calculation*************************//
    //matrix_temp1 = ATA
    for(int i=0; i<n_poly; i++)
    {
        for(int j=0; j<n_poly; j++)
        {
            matrix_temp1[i][j]=0;
            for(int k=0; k<x.size(); k++)
                matrix_temp1[i][j] += matrix_T[i][k]*matrix[k][j];
        }
    }
    
    matrix_copy(matrix_temp1, matrix_ATA, n_poly); 

    /*
    printf("ATA\n");
    for(int i=0; i < n_poly; i++)
    {
        for(int j=0; j < n_poly; j++)
        {
            printf("%lf ", matrix_temp1[i][j]);
        }
        printf("\n");
    }
    */

    //matrix_temp1 is ATA+lambda*I
    for(int i=0; i<n_poly; i++)
    {
        matrix_temp1[i][i]+=lambda;
    }


    //LU Decomposition, matrix_temp2 is U and matrix_temp3 is L
    matrix_LU(matrix_temp1, matrix_temp2, matrix_temp3, n_poly);

    //matrix_temp1 = U-1
    double s;
    matrix_reset(matrix_temp1, n_poly);
    for (int i=0; i<n_poly; i++) //U inv
	{
		matrix_temp1[i][i]=1/matrix_temp2[i][i];//對角元素直接倒數
		for (int k=i-1; k>=0; k--)
		{
			s=0;
			for (int j=k+1; j<=i; j++)
				s=s+matrix_temp2[k][j]*matrix_temp1[j][i];
			matrix_temp1[k][i]=-s/matrix_temp2[k][k];//迭代计算，按列倒序依次得到每一个值，
		}
	}


    //matrix_temp2 = L-1
    matrix_reset(matrix_temp2, n_poly);
	for (int i=0; i<n_poly; i++) //L inv
	{
		matrix_temp2[i][i]=1/matrix_temp3[i][i]; //對角元素直接倒數
		for (int k=i+1; k<n_poly; k++)
		{
			for (int j=i;j<=k-1;j++)
				matrix_temp2[k][i]=matrix_temp2[k][i]-matrix_temp3[k][j]*matrix_temp2[j][i];   //迭代计算，按列顺序依次得到每一个值
		}
	}


    //matrix_temp1 = U-1, matrix_temp2 = L-1, matrix_temp3 = (ATA+lambda I)-1
    matrix_mul(matrix_temp1, matrix_temp2, matrix_temp3, n_poly);


    // ATb
    for(int i=0; i<n_poly; i++)
    {
        ATb.push_back(0);
        for(int j=0; j<x.size(); j++)
        {
            ATb[i]+=matrix_T[i][j]*b[j];
        }
    }

    /*
    printf("ATb\n");
    for(int i=0; i<ATb.size(); i++)
    {
        printf("%lf ", y_cal[i]);
    }
    printf("\n");
    */

    //**************************3.closed_form_LSE*************************//
    /////////coeff calculation/////////
    for(int i=0; i<n_poly; i++)
    {   
        double s=0;
        for(int j=0; j<n_poly; j++)
        {
            s += matrix_temp3[i][j] * ATb[j];
        }
        coeff.push_back(s);
    }

    //Print Fitting Line
    printf("LSE: \n");
    printf("Fitting Line: ");
    for(int i=n_poly-1; i>=0; i--)
    {
        if(i>0) printf("%lfx^%d + ", coeff[n_poly-1-i], i);
        else printf("%lf\n", coeff[n_poly-1]);
    }
    
    //Print Total Error
    double Error = 0; 
    for(int i=0; i<b.size(); i++)
    {
        double sum = 0;
        for(int j=n_poly-1; j>=0; j--)
        {
            sum += coeff[n_poly-1-j]*pow(x[i], j);
            
        }
        Error += pow(sum-b[i], 2);
    }
    printf("Total Error: %lf\n", Error);
    printf("\n");
    matrix_reset(matrix_temp1, n_poly);
    matrix_reset(matrix_temp2, n_poly);
    matrix_reset(matrix_temp3, n_poly);
    coeff.clear();


    //**************************4.Steepest Descent*************************//
    coeff.reserve(n_poly);
    coeff_temp1.reserve(n_poly);
    coeff_temp2.reserve(n_poly);
    coeff_temp3.reserve(n_poly);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution <double> dist(-100, 100);

    for(int i=0; i<n_poly; i++)
    {
        coeff_1.push_back(dist(gen));
    }

    double epsilon = 10000;
    double epsilon_max = 1e-6;

    int Iter = 0;
    int maxIterations = 100;

    while(epsilon > epsilon_max && Iter < 100)
    {
        //Xn = Xn-1 - learning rate * ( 2(ATAx-ATb) + lambda*sign(x) )
        matrix_mul_vector(matrix_ATA, coeff_1, coeff_temp1, n_poly, n_poly);
        matrix_mul_vector(matrix_T, b, coeff_temp2, n_poly, x.size());
        vector_minus(coeff_temp1, coeff_temp2, coeff_temp3, n_poly);

        for(int i=0; i<n_poly; i++)
        {
            coeff_temp3[i]*=2;
            if(coeff_1[i]>0) coeff_temp3[i] += lambda;
            else if(coeff_1[i]<0) coeff_temp3[i] -= lambda;
            coeff_temp3[i]*=0.00001;
        }

        vector_minus(coeff_1, coeff_temp3, coeff, n_poly);
        

        //Error
        Error = 0;
        for(int i=0; i<b.size(); i++)
        {
            double sum = 0;
            for(int j=n_poly-1; j>=0; j--)
            {
                sum += coeff[n_poly-1-j]*pow(x[i], j);
                
            }
            Error += pow(sum-b[i], 2);
        }

        //loss
        epsilon = 0;
        for(int i=0; i<n_poly; i++)
        {
            epsilon += pow(coeff[i]-coeff_1[i], 2);
        }

        for(int i=0; i<n_poly; i++)
        {
            coeff_1[i]=coeff[i];
        }

        Iter++;
    }


    //Print Fitting Line
    printf("Steepest Descent: \n");
    printf("Fitting Line: ");
    for(int i=n_poly-1; i>=0; i--)
    {
        if(i>0) printf("%lfx^%d + ", coeff[n_poly-1-i], i);
        else printf("%lf\n", coeff[n_poly-1]);
    }
    printf("Total Error: %lf\n", Error);
    printf("\n");
    coeff.clear();
    coeff_1.clear();
    coeff_temp1.clear();
    coeff_temp2.clear();
    coeff_temp3.clear();


    //**************************5.Newtons Method*************************//
    coeff.reserve(n_poly);
    coeff_temp1.reserve(n_poly);
    coeff_temp2.reserve(n_poly);
    coeff_temp3.reserve(n_poly);


    for(int i=0; i<n_poly; i++)
    {
        coeff_1.push_back(dist(gen));
    }

    epsilon = 100;
    epsilon_max = 1e-6;

    Iter = 0;
    maxIterations = 100;
    
    //LU Decomposition, matrix_temp2 is U and matrix_temp3 is L
    matrix_LU(matrix_ATA, matrix_temp2, matrix_temp3, n_poly);

    //matrix_temp1 = U-1
    s=0;
    matrix_reset(matrix_temp1, n_poly);
    for (int i=0; i<n_poly; i++) //U inv
	{
		matrix_temp1[i][i]=1/matrix_temp2[i][i];//對角元素直接倒數
		for (int k=i-1; k>=0; k--)
		{
			s=0;
			for (int j=k+1; j<=i; j++)
				s=s+matrix_temp2[k][j]*matrix_temp1[j][i];
			matrix_temp1[k][i]=-s/matrix_temp2[k][k];//迭代计算，按列倒序依次得到每一个值，
		}
	}


    //matrix_temp2 = L-1
    matrix_reset(matrix_temp2, n_poly);
	for (int i=0; i<n_poly; i++) //L inv
	{
		matrix_temp2[i][i]=1/matrix_temp3[i][i]; //對角元素直接倒數
		for (int k=i+1; k<n_poly; k++)
		{
			for (int j=i;j<=k-1;j++)
				matrix_temp2[k][i]=matrix_temp2[k][i]-matrix_temp3[k][j]*matrix_temp2[j][i];   //迭代计算，按列顺序依次得到每一个值
		}
	}


    //matrix_temp1 = U-1, matrix_temp2 = L-1, matrix_temp3 = (ATA)-1
    matrix_mul(matrix_temp1, matrix_temp2, matrix_temp3, n_poly);
    matrix_copy(matrix_temp3, matrix_ATA_inv, n_poly);

    while(epsilon > epsilon_max && Iter < 100)
    {
        //Xn = Xn-1 - (A.TA)^(-1) * (A.TA(Xn-1) - A.Tb)
        matrix_mul_vector(matrix_ATA, coeff_1, coeff_temp2, n_poly, n_poly);
        vector_minus(coeff_temp2, ATb, coeff_temp3, n_poly);
        matrix_mul_vector(matrix_ATA_inv, coeff_temp3, coeff_temp1, n_poly, n_poly);
        vector_minus(coeff_1, coeff_temp1, coeff, n_poly);
        

        //Error
        Error = 0;
        for(int i=0; i<b.size(); i++)
        {
            double sum = 0;
            for(int j=n_poly-1; j>=0; j--)
            {
                sum += coeff[n_poly-1-j]*pow(x[i], j);
                
            }
            Error += pow(sum-b[i], 2);
        }

        //loss
        epsilon = 0;
        for(int i=0; i<n_poly; i++)
        {
            epsilon += pow(coeff[i]-coeff_1[i], 2);
        }

        ///*
        for(int i=0; i<n_poly; i++)
        {
            coeff_1[i]=coeff[i];
        }
        //*/
        Iter++;
    }
    Iter = 0;

    //Print Fitting Line
    printf("Newton's Method: \n");
    printf("Fitting Line: ");
    for(int i=n_poly-1; i>=0; i--)
    {
        if(i>0) printf("%lfx^%d + ", coeff[n_poly-1-i], i);
        else printf("%lf\n", coeff[n_poly-1]);
    }
    printf("Total Error: %lf\n", Error);
    coeff.clear();
    coeff_1.clear();
    coeff_temp1.clear();
    coeff_temp2.clear();
    coeff_temp3.clear();

    //**************************6.Dynamic Array Release*************************//
    for(int i = 0; i < n_poly; i++)
        delete [] matrix[i];
    delete [] matrix;

    for(int i = 0; i < n_poly; i++)
        delete [] matrix_temp1[i];
    delete [] matrix_temp1;
    

    for(int i = 0; i < n_poly; i++)
        delete [] matrix_temp2[i];
    delete [] matrix_temp2;

    for(int i = 0; i < n_poly; i++)
        delete [] matrix_temp3[i];
    delete [] matrix_temp3;

    for(int i = 0; i < n_poly; i++)
        delete [] matrix_ATA[i];
    delete [] matrix_ATA;

    for(int i = 0; i < n_poly; i++)
        delete [] matrix_ATA_inv[i];
    delete [] matrix_ATA_inv;


    system("pause");
    return 0;
}


void read_file()
{
    ifstream in;
    in.open("testfile.txt");

    if(in.fail()){ 
        printf("input file opening failed");
        exit(1); 
    }
    else printf("successful openning\n");

    while(in)
    {
        string s;
        if(!getline(in, s)) break;
        stringstream ss(s);
        string str;

        while (getline(ss, str, ','))
        {
            infile.push_back(stof(str));
        }
        ss.clear();
    }
    in.close();
}

void matrix_transpose(double **matrix1, double **matrix2, int n)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            matrix2[i][j] = matrix1[j][i];
        }
    }
}

void matrix_mul(double **matrix1, double **matrix2, double **matrix3, int n)
{

    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            matrix3[i][j] = 0;
            for(int k=0; k<n; k++)
            {
                matrix3[i][j] += matrix1[i][k]*matrix2[k][j];
            }
        }
    }
}

void matrix_LU(double **matrix1, double **matrix2, double **matrix3, int n)
{
    //Copy matrix1 to matrix2
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            matrix2[i][j] = matrix1[i][j];
        }
    }

    //U
    int flag=0;
    for(int j=0; j<n; j++)
    {
        for(int i=j+1; i<n; i++)
        {
            LU_parameter.push_back(matrix2[i][j]/matrix2[j][j]);
            for(int k=0; k<n; k++)
            {
                //printf("matrix[%d][%d]-=matrix[%d][%d]*%lf\n",i ,k , i, j, matrix1[i][j]/matrix1[j][j]);
                matrix2[i][k] -= matrix2[j][k]*LU_parameter[flag];
            }
            flag++;
        }
    }
    //printf("Size of LU_parameter=%d\n", LU_parameter.size());
    
    int vector_point = 0;

    //L
    for(int j=0; j<n; j++)
    {
        for(int i=0; i<n; i++)
        {
            if(i==j) matrix3[i][j] = 1;
            else if(i>j)
            {
                matrix3[i][j] = LU_parameter[vector_point];
                vector_point++;
            }
            else matrix3[i][j] = 0;
        }
    }

    LU_parameter.clear();
}

void matrix_copy(double **matrix1, double **matrix2, int n)    //Copy matrix1 to matrix2
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            matrix2[i][j] = matrix1[i][j];
        }
    }
}

void matrix_reset(double **matrix, int n)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            matrix[i][j] = 0;
        }
    }
}

void vector_minus(vector <double> &vec1, vector <double> &vec2, vector <double> &vec3, int n) //vec1 - vec2 = vec3
{
    for(int i=0; i<n; i++)
    {
        vec3[i] = vec1[i] - vec2[i];
    }
}

void matrix_mul_vector(double **matrix, vector <double> &vec1, vector <double> &vec2, int n, int size) //matrix*vec1=vec2, matrix is n*size
{
    for(int i=0; i<n; i++)
    {
        vec2[i] = 0;
        for(int j=0; j<size; j++)
        {
            vec2[i]+=matrix[i][j]*vec1[j];
        }
    }
}

/////////////////TESTING///////////////////
    //print ATA+lambda*I
    /*
    printf("Print ATA+lambda*I\n");
    for(int i=0; i < n_poly; i++)
    {
        for(int j=0; j < n_poly; j++)
        {
            printf("%lf ", matrix_temp2[i][j]);
        }
        printf("\n");
    }

    //print U
    printf("Print U\n");
    for(int i=0; i < n_poly; i++)
    {
        for(int j=0; j < n_poly; j++)
        {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
    //print L
    printf("Print L\n");
    for(int i=0; i < n_poly; i++)
    {
        for(int j=0; j < n_poly; j++)
        {
            printf("%lf ", matrix_temp1[i][j]);
        }
        printf("\n");
    }
    */