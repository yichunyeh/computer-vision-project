#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#define DEG2RAD 0.017453293f
#define PI 3.1415926

using namespace cv;
using namespace std;
class find_parallelograms{
    private:
        Mat angles; //Angle map
	int img_w;  //image width
        int img_h;  //image height
	unsigned int* accu;   //accumulator
        int accu_w; //accumulator width
        int accu_h; //accumulator height

    public:
	Mat image;
        Mat gray(Mat); //Grayscale
        vector<vector<double> > createFilter(int, int, double); //Creates a gaussian filter
        Mat useFilter(Mat, vector<vector<double> >); //Use gaussian filter
        Mat sobel(Mat); //Sobel filtering
        Mat nonMaxSupp(Mat); //Non-maxima supp.
	Mat threshold(Mat, int, int); //Double threshold and finalize picture
	void PrintLines(int, int);
	int HoughTransform(unsigned char*, int, int);
        //int HoughTransform(Mat, int, int);
	vector< pair< pair<int, int>, pair<int, int> > > GetLines(int);
	const unsigned int* GetAccu(int *w, int *h);
};

Mat find_parallelograms::gray(Mat image){
    Mat gray_image(image.rows, image.cols, CV_8UC1);
    for(int i=0; i<image.rows; i++){
        for(int j=0; j<image.cols; j++){
            gray_image.at<uchar>(i,j) = 0.03*image.at<Vec3b>(i,j)[0]+ 0.59*image.at<Vec3b>(i,j)[1]+ 0.11*image.at<Vec3b>(i,j)[2];
        }
    }
    return gray_image;
}

vector<vector<double> > find_parallelograms::createFilter(int row, int column, double sigmaIn){
    vector<vector<double> > filter;

    for (int i = 0; i < row; i++){
        vector<double> col;
        for (int j = 0; j < column; j++){
            col.push_back(-1);
        }
	filter.push_back(col);
    }

    float coordSum = 0;
    float constant = 2.0 * sigmaIn * sigmaIn;

    // Sum is for normalization
    float sum = 0.0;

    for (int x = - row/2; x <= row/2; x++){
	for (int y = -column/2; y <= column/2; y++){
		coordSum = (x*x + y*y);
		filter[x + row/2][y + column/2] = (exp(-(coordSum) / constant)) / (M_PI * constant);
		sum += filter[x + row/2][y + column/2];
	}
    }

    // Normalize the Filter
    for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            filter[i][j] /= sum;

    return filter;
}

Mat find_parallelograms::useFilter(Mat img_in, vector<vector<double> > filterIn){
    int size = (int)filterIn.size()/2;
    Mat filteredImg = Mat(img_in.rows - 2*size, img_in.cols - 2*size, CV_8UC1);
    for (int i = size; i < img_in.rows - size; i++){
	for (int j = size; j < img_in.cols - size; j++){
		double sum = 0;
    
		for (int x = 0; x < filterIn.size(); x++)
			for (int y = 0; y < filterIn.size(); y++){
	                    sum += filterIn[x][y] * (double)(img_in.at<uchar>(i + x - size, j + y - size));
			}
    
            filteredImg.at<uchar>(i-size, j-size) = sum;
	}

    }    
    return filteredImg;
}

Mat find_parallelograms::sobel(Mat gFiltered){

    //Sobel X Filter
    double x1[] = {-1.0, 0, 1.0};
    double x2[] = {-2.0, 0, 2.0};
    double x3[] = {-1.0, 0, 1.0};

    vector< vector<double> > xFilter(3);
    xFilter[0].assign(x1, x1+3);
    xFilter[1].assign(x2, x2+3);
    xFilter[2].assign(x3, x3+3);
    
    //Sobel Y Filter
    double y1[] = {1.0, 2.0, 1.0};
    double y2[] = {0, 0, 0};
    double y3[] = {-1.0, -2.0, -1.0};
    
    vector< vector<double> > yFilter(3);
    yFilter[0].assign(y1, y1+3);
    yFilter[1].assign(y2, y2+3);
    yFilter[2].assign(y3, y3+3);
    
    //Limit Size
    int size = (int)xFilter.size()/2;
    
    Mat filteredImg = Mat(gFiltered.rows - 2*size, gFiltered.cols - 2*size, CV_8UC1);
    
    angles = Mat(gFiltered.rows - 2*size, gFiltered.cols - 2*size, CV_32FC1); //AngleMap

    for (int i = size; i < gFiltered.rows - size; i++){
	for (int j = size; j < gFiltered.cols - size; j++){
	    double sumx = 0;
	    double sumy = 0;

	    for (int x = 0; x < xFilter.size(); x++){
		for (int y = 0; y < xFilter.size(); y++){
		    sumx += xFilter[x][y] * (double)(gFiltered.at<uchar>(i + x - size, j + y - size)); //Sobel_X Filter Value
		    sumy += yFilter[x][y] * (double)(gFiltered.at<uchar>(i + x - size, j + y - size)); //Sobel_Y Filter Value
		}
	    }
		double sumxsq = sumx*sumx;
		double sumysq = sumy*sumy;
    
	    double sq2 = sqrt(sumxsq + sumysq);
	    
	    if(sq2 > 255) //Unsigned Char Fix
		sq2 =255;
	    filteredImg.at<uchar>(i-size, j-size) = sq2;
 
	    if(sumx==0) //Arctan Fix
		angles.at<float>(i-size, j-size) = 90;
	    else
		angles.at<float>(i-size, j-size) = atan(sumy/sumx);
	}
    }
    
    return filteredImg;
}
Mat find_parallelograms::nonMaxSupp(Mat sFiltered){
    Mat nonMaxSupped = Mat(sFiltered.rows-2, sFiltered.cols-2, CV_8UC1);
    for (int i=1; i< sFiltered.rows - 1; i++){
        for (int j=1; j<sFiltered.cols - 1; j++){
            float Tangent = angles.at<float>(i,j);

            nonMaxSupped.at<uchar>(i-1, j-1) = sFiltered.at<uchar>(i,j);
            //Horizontal Edge
            if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5))){
                if ((sFiltered.at<uchar>(i,j) < sFiltered.at<uchar>(i,j+1)) || (sFiltered.at<uchar>(i,j) < sFiltered.at<uchar>(i,j-1)))
                    nonMaxSupped.at<uchar>(i-1, j-1) = 0;
            }
            //Vertical Edge
            if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5))){
                if ((sFiltered.at<uchar>(i,j) < sFiltered.at<uchar>(i+1,j)) || (sFiltered.at<uchar>(i,j) < sFiltered.at<uchar>(i-1,j)))
                    nonMaxSupped.at<uchar>(i-1, j-1) = 0;
            }
            
            //-45 Degree Edge
            if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5))){
                if ((sFiltered.at<uchar>(i,j) < sFiltered.at<uchar>(i-1,j+1)) || (sFiltered.at<uchar>(i,j) < sFiltered.at<uchar>(i+1,j-1)))
                    nonMaxSupped.at<uchar>(i-1, j-1) = 0;
            }
            
            //45 Degree Edge
            if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5))){
                if ((sFiltered.at<uchar>(i,j) < sFiltered.at<uchar>(i+1,j+1)) || (sFiltered.at<uchar>(i,j) < sFiltered.at<uchar>(i-1,j-1)))
                    nonMaxSupped.at<uchar>(i-1, j-1) = 0;
            }
        }
    }
    return nonMaxSupped;
}

Mat find_parallelograms::threshold(Mat imgin,int low, int high){
    if(low > 255)
        low = 255;
    if(high > 255)
        high = 255;
    
    Mat EdgeMat = Mat(imgin.rows, imgin.cols, imgin.type());
    
    for (int i=0; i<imgin.rows; i++){
        for (int j = 0; j<imgin.cols; j++){
            EdgeMat.at<uchar>(i,j) = imgin.at<uchar>(i,j);
            if(EdgeMat.at<uchar>(i,j) > high)
                EdgeMat.at<uchar>(i,j) = 255;
            else if(EdgeMat.at<uchar>(i,j) < low)
                EdgeMat.at<uchar>(i,j) = 0;
            else{
                bool anyHigh = false;
                bool anyBetween = false;
                for (int x=i-1; x < i+2; x++){
                    for (int y = j-1; y<j+2; y++){
                        if(x <= 0 || y <= 0 || EdgeMat.rows || y > EdgeMat.cols) //Out of bounds
                            continue;
                        else{
                            if(EdgeMat.at<uchar>(x,y) > high){
                                EdgeMat.at<uchar>(i,j) = 255;
                                anyHigh = true;
                                break;
                            }
                            else if(EdgeMat.at<uchar>(x,y) <= high && EdgeMat.at<uchar>(x,y) >= low)
                                anyBetween = true;
                        }
                    }
                    if(anyHigh)
                        break;
                }
                if(!anyHigh && anyBetween)
                    for (int x=i-2; x < i+3; x++){
                        for (int y = j-1; y<j+3; y++){
                            if(x < 0 || y < 0 || x > EdgeMat.rows || y > EdgeMat.cols) //Out of bounds
                                continue;
                            else{
                                if(EdgeMat.at<uchar>(x,y) > high){
                                    EdgeMat.at<uchar>(i,j) = 255;
                                    anyHigh = true;
                                    break;
                                }
                            }
                        }
                        if(anyHigh)
                            break;
                    }
                if(!anyHigh)
                    EdgeMat.at<uchar>(i,j) = 0;
            }
        }
    }
    return EdgeMat;
}

int find_parallelograms::HoughTransform(unsigned char* img_data, int w, int h){  
    img_w = w;  
    img_h = h;  

    //Create the accu  
    double hough_h = ((sqrt(2.0) * (double)(h>w?h:w)) / 2.0);  
    accu_h = hough_h * 2.0; // -r -> +r  
    accu_w = 180;  

    accu = (unsigned int*)calloc(accu_h*accu_w, sizeof(unsigned int));  
    //accu = Mat(accu_h, accu_w, CV_8UC1);


    double center_x = w/2;  
    double center_y = h/2;  

    for(int y=0; y<h; y++){  
        for(int x=0; x<w; x++){  
            if(img_data[(y*w) + x] > 250){  
                for(int t=0; t<180; t++){  
                    double r = ( ((double)x-center_x) * cos((double)t*PI/180.0)) + (((double)y-center_y) * sin((double)t*PI/180.0));  
                    accu[(int)((round(r+hough_h) * 180.0)) + t]++;  
                 }  
             }  
         }  
     }  
     return 0;  
}
void find_parallelograms::PrintLines(int t, int r){
	vector< pair< pair<int, int>, pair<int, int> > > tmp;
	int x1, y1, x2, y2;  
	x1 = y1 = x2 = y2 = 0;  
			
	if(t >= 45 && t <= 135){  
	    //y = (r - xcos(t)) / sin(t)  
	    x1 = 0;  
	    //y1 = ((double)(r-(accu_h/2)) - ((x1-(img_w/2))*cos(t*PI/180.0))) / sin(t*PI/180.0) + (img_h/2);  
	    y1 = ((double)(r-(accu_h/2)) - ((x1-(img_w/2))*cos(t*DEG2RAD))) / sin(t*DEG2RAD) + (img_h/2); 
	    x2 = img_w - 0;  
	    y2 = ((double)(r-(accu_h/2)) - ((x2-(img_w/2))*cos(t*DEG2RAD))) / sin(t*DEG2RAD) + (img_h/2);  
	    //y1 = y1>y2 ? y1:y2;
	    //for(int i=x1; i<x2; i++){
		//for(int j=y1; j<y2; j++){
		   //accu[]
		//}
	    //}
	}  
	else{  
	    //x = (r - y sin(t)) / cos(t);  
	    y1 = 0;  
	    x1 = ((double)(r-(accu_h/2)) - ((y1-(img_h/2))*sin(t*DEG2RAD))) / cos(t*DEG2RAD)+(img_w/2);  
	    y2 = img_h - 0;  
	    x2 = ((double)(r-(accu_h/2)) - ((y2-(img_h/2))*sin(t*DEG2RAD))) / cos(t*DEG2RAD)+(img_w/2);  
	} 
     
	line(image, Point(x1, y1), Point(x2, y2), Scalar(0, 225, 0), 5, 8);

	//determine the paralle lines here
	//tmp.push_back(pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1,y1), pair<int, int>(x2,y2))); 
	//para_lines[accu_w] = pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1,y1), pair<int, int>(x2,y2));
	//return tmp;
}

//void find_parallelograms::PrintShape(int t, int r, ){
//	vector< pair< pair<int, int>, pair<int, int> > > tmp;
//	int x1, y1, x2, y2;  
//	x1 = y1 = x2 = y2 = 0;  
//			
//	if(t >= 45 && t <= 135){  
//	    //y = (r - xcos(t)) / sin(t)  
//	    x1 = 0;  
//	    //y1 = ((double)(r-(accu_h/2)) - ((x1-(img_w/2))*cos(t*PI/180.0))) / sin(t*PI/180.0) + (img_h/2);  
//	    y1 = ((double)(r-(accu_h/2)) - ((x1-(img_w/2))*cos(t*DEG2RAD))) / sin(t*DEG2RAD) + (img_h/2); 
//	    x2 = img_w - 0;  
//	    y2 = ((double)(r-(accu_h/2)) - ((x2-(img_w/2))*cos(t*DEG2RAD))) / sin(t*DEG2RAD) + (img_h/2);  
//	    //y1 = y1>y2 ? y1:y2;
//	    //for(int i=x1; i<x2; i++){
//		//for(int j=y1; j<y2; j++){
//		   //accu[]
//		//}
//	    //}
//	}  
//	else{  
//	    //x = (r - y sin(t)) / cos(t);  
//	    y1 = 0;  
//	    x1 = ((double)(r-(accu_h/2)) - ((y1-(img_h/2))*sin(t*DEG2RAD))) / cos(t*DEG2RAD)+(img_w/2);  
//	    y2 = img_h - 0;  
//	    x2 = ((double)(r-(accu_h/2)) - ((y2-(img_h/2))*sin(t*DEG2RAD))) / cos(t*DEG2RAD)+(img_w/2);  
//	} 
//     
//	line(image, Point(x1, y1), Point(x2, y2), Scalar(0, 225, 0), 5, 8);
//
//	//determine the paralle lines here
//	//tmp.push_back(pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1,y1), pair<int, int>(x2,y2))); 
//	//para_lines[accu_w] = pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1,y1), pair<int, int>(x2,y2));
//	//return tmp;
//}

vector< pair< pair<int, int>, pair<int, int> > > find_parallelograms::GetLines(int threshold){  
    vector<int> lines;
    //vector< pair< pair<int, int>, pair<int, int> > > lines;
    lines.push_back(0);
    lines.resize(0);

    vector< pair<int, vector<int> > >  para_lines;
    //vector<vector< pair< pair<int, int>, pair<int, int> > > > para_lines;  //To store the parallel lines.

    vector< pair< pair<int, int>, pair<int, int> > > shape_lines; //To store the lines of the shape.

    //accu_point = (unsigned int*)calloc(img_h*img_w, sizeof(unsigned int));  

    //vector< int, vector< pair<int, int>, pair<int, int> > paralle;
    if(accu == 0)  
        return shape_lines;  
    //int cnt; //To count the parallel line.
    for(int t=0; t<accu_w; t++){  
	//cnt = 0; //Initialize the counter to 0 when move to another degree.
	//para_lines[accu_w] = pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1,y1), pair<int, int>(x2,y2));

	//!!!make sure to chage the print initial value!!!! since I initial the first one to (-1,-1)
	//lines.push_back(pair< pair<int, int>, pair<int, int> >(pair<int, int>(-1,-1), pair<int, int>(-1,-1)));
        
        //need to check other places!
	//lines.push_back(-1);

	//cout << "initial: " << lines[0] << endl;
        for(int r=0; r<accu_h; r++){  
            if((int)accu[(r*accu_w) + t] >= threshold){  
                //cnt++;
		//Is this point a local maxima (9x9)  
                int max = accu[(r*accu_w) + t];  
                for(int ly=-4; ly<=4; ly++){  
                    for(int lx=-4; lx<=4; lx++){  
                        if((ly+r>=0 && ly+r<accu_h) && (lx+t>=0 && lx+t<accu_w)){  
                            if((int)accu[((r+ly)*accu_w) + (t+lx)] > max){  
                                max = accu[((r+ly)*accu_w) + (t+lx)];  
                                ly = lx = 5;  
                            }  
                        }  
                    }  
                }  
                if(max > (int)accu[(r*accu_w) + t])  
                    continue;  
		//if(cnt % 2 == 0)
		    //continue;
                
		//int value = lines.begin();
		//cout << "before push_back r: " << lines[t] << endl;
		lines.push_back(r);
	        //cout << "after push_back r: " << lines[t] << " And r is: " << r << endl;
                //lines.push_back(pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1,y1), pair<int, int>(x2,y2)));  
		
            } 
        } 
	vector<int>::iterator itL;
	cout << "lines' size: " << lines.size() << endl;
	cout << "R's at angle: " << t << " is: ";
	for(itL=lines.begin(); itL!=lines.end(); itL++){
	    cout << *itL << " ";
	}
	cout << endl;
	//fine tune here: put some different but similar degree together
	if(t%10!=0){
	    if(t==179){
	        para_lines.push_back(pair<int, vector <int> >(t, vector <int> (lines)));
		cout << "t=179!" << endl;
	        cout << "size of line " << t << ":" << para_lines.size() << endl;
		lines.resize(0);
	    }
	    continue;
	    //vector<int> para_lines_tmp(-1);
	    //para_lines.push_back(
	}
        para_lines.push_back(pair<int, vector <int> >(t, vector <int> (lines)));
	cout << "size of line " << t << ":" << para_lines.size() << endl;
	lines.resize(0);
	//cout << "size of line " << t << "(after resize):" << lines.size() << endl;

    }
    cout << "push finish." << endl;

    //check if it has pair
    vector< pair <int, vector <int> > > final_para_lines;
    for(int i=0; i<para_lines.size(); i++){
        if(para_lines.at(i).second.size()>=2)
	    final_para_lines.push_back(para_lines.at(i));
    }

    vector< pair <int, vector <int> > >::iterator L;
    //cout << "lines' size: " << lines.size() << endl;
    //cout << "Points in final_para_lines: " << i << " is: ";
    cout << "Number of lines in final_para_lines: " << final_para_lines.size() << endl;
    for(L=final_para_lines.begin(); L!=final_para_lines.end(); L++){
	PrintLines(L->first, L->second.at(0));
	PrintLines(L->first, L->second.at(1));
        cout << "angle: " << L->first << "  first r: " << L->second.at(0) << " second r: " << L->second.at(1);
	cout << endl;
    }	
    cout << endl;

    
    
    vector< vector <int> >::iterator t;
    //vector< pair< pair<int, int>, pair<int, int> > >::iterator t; 
    int inter_x1; //first intersection coordinates x
    int inter_y1; //first intersection coordinates y
    //int inter_x2; //second intersection coordinates x
    //int inter_y2; //second intersection coordinates y
    int d1, d2, r1, r2;
    d1 = d2 = r1 = r2 = 0;

    vector< pair <int, int> > points;

    //for(int i=0; i<accu_w; i++){ //or we can use vector.size()
    for(int i=0; i<final_para_lines.size(); i++){ //need to check here!
        //cout << "accu_w" << i << endl;
        //cout << "point1: " << Point(para_lines.at(i).back().first.first, para_lines.at(i).back().first.second) << " point2:" << Point(para_lines.at(i).back().second.first, para_lines.at(i).back().second.second) << endl;
	//if(para_lines.at(i).back().first.first ==(-1) && para_lines.at(i).back().first.second==(-1) && para_lines.at(i).back().second.first ==(-1) && para_lines.at(i).back().second.second ==(-1))
	cout<<"size of final_para_lines.at: "<< i << " is: " << final_para_lines.at(i).second.size() << endl;
	//this part later can skip
	//if(para_lines.at(i).size()<3)
	//    continue;
 	//if(para_lines.at(i).size()>=3){  //(need to add one because I add one more line for initializing the vector
	//    cout << "larger than3" << endl;
	//    //bool tmp = para_lines.at(i).size() >= 3 ? true:
	//    if(para_lines.at(i+90).size() >= 3){
	//	//cout << "prara_line.at(i).size(): " << para_lines.at(i).size() << " para_lines.at(i+90).size(): " << para_lines.at(i+90).size() <<endl;
	for(int j=0; j<final_para_lines.size(); j++){
		//put points into lines
		points.resize(0);
		for(int a=0; a<final_para_lines.at(i).second.size(); a++){
		    for(int b=0; b<final_para_lines.at(j).second.size(); b++){
			if(final_para_lines.at(i).first!=final_para_lines.at(j).first){
				cout << "Now is for line: " << i << " and line: " << j <<endl;
				cout <<"Intersection point: " << a << " and intersection point: " << b << endl;
				d1 = final_para_lines.at(i).first;
				d2 = final_para_lines.at(j).first;
				cout << "d1(before): " << d1 << " d2(before): " << d2 << endl;
				//d1 = final_para_lines.at(i).first*DEG2RAD;
				//d2 = final_para_lines.at(j).first*DEG2RAD;
				//r1 = final_para_lines.at(i).second.at(a)-(accu_h/2);
				//r2 = final_para_lines.at(j).second.at(b)-(accu_h/2);
				r1 = (double)final_para_lines.at(i).second.at(a);
				r2 = (double)final_para_lines.at(j).second.at(b);
				cout << "d1: " << d1 << " d2: " << d2 << " r1: " << r1 << " r2: " << r2 << endl;
				//find the intersection points
				//y1
				//x1 = ((double)(r-(accu_h/2)) - ((y1-(img_h/2))*sin(t*DEG2RAD))) / cos(t*DEG2RAD)+(img_w/2);
				inter_x1 = (cos(d2)*r1-cos(d1)*r2) / (sin(d1)*cos(d2)-cos(d1)*sin(d2));
				inter_y1 = (r1-inter_y1*sin(d1)) / cos(d1);
				//inter_x1 = (double)(sin(d1)*r2-sin(d2)*r1) / (sin(d1)*cos(d2)-sin(d2)*cos(d1))+img_h/2;
				//inter_y1 = (double)(cos(d1)*r2-cos(d2)*r1) / (cos(d1)*sin(d2)-sin(d1)*cos(d2))+img_w/2;
				points.push_back(pair<int, int>(inter_x1, inter_y1));
				vector< pair <int, int> >::iterator itP;
				//cout << "lines' size: " << itP.size() << endl;
				cout << "Points in final_para_lines: " << i << " is: " << endl;
				for(itP=points.begin(); itP!=points.end(); itP++){
				    cout << "point " << itP->first << " " << itP->second;
				    cout << endl;
				}
				cout << endl;
				if(points.size()==2){
				    shape_lines.push_back(pair< pair<int, int>, pair<int, int> >(points.at(0), points.at(1)));
				    points.resize(0);
				}
			}
			//inter_x2 = (cos(i)*para_lines[i+90][b+1]-cos(i+90)*para_lines[i][a]) / (cos(i)*sin(i+90)-sin(i)*cos(i+90)); 
			//inter_y2 = (cos(i)*para_lines[i+90][b+1]-sin(i+90)*para_lines[i][a]) / (cos(i+90)*sin(i)-sin(i+90)*cos(i));
			//shape_lines.push_back(pair< pair<int, int>, pair<int, int> >(pair<int, int>(inter_x1,inter_y1), pair<int, int>(inter_x2,inter_y2)));
			//inter_point.push_back(); 
		    }
		    //shape_lines.push_back(pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1,y1), pair<int, int>(x2,y2)));
	        }
	  //  }
	}    	    
	    //cout << "create iterator!" << endl;
	    //cout << "Parallel lines in theta: " << i << endl;
	    //cout << "size of the para_lines: " << para_lines.at(i).size() << endl;
	    //t=para_lines.at(i).begin();
	    ////for(it=lines.begin(); it!=lines.end(); it++){  
	    //for(advance(t, 1); t!=para_lines.at(i).end(); t++){
	        ////here need to modify! since I change vector<pair...> to vector<vector<int> >	
	        //cout << "point1: " << Point(t->first.first, t->first.second) << " point2:" << Point(t->second.first, t->second.second) << endl; 
	        ////line(image, Point(it->first.first, it->first.second), Point(it->second.first, it->second.second), Scalar(0, 0, 255), 5, 8);
	    //}      

        //para_lines[accu_w]
	//cout << "Parallel lines in theta: " << accu_w << endl;
	//cout << "point1: " << Point(it->first.first, it->first.second) << " point2:" << Point(it->second.first, it->second.second) << endl; 

	 
    }

    cout << "shape_lines' size: " << shape_lines.size() << "  threshold: " << threshold << endl;  
    return shape_lines;  
} 

//We don't use this function right now.
const unsigned int* find_parallelograms::GetAccu(int *w, int *h)
{
    *w = accu_w;
    *h = accu_h;

    return accu;
}
   
int main(int argc, char** argv)
{
    //char* imageName = argv[1];
    int num;
    //Mat image;
    find_parallelograms fp;
    cout << "Using image(1,2 or3):";
    cin >> num;

    if(num==1)
    	fp.image = imread("TestImage1c.jpg", 1);
    else if(num==2)
        fp.image = imread("TestImage2c.jpg", 1);
    else if(num==3)
	fp.image = imread("TestImage3.jpg", 1);

    //imwrite("Gray_Image.jpg", gray_image);

    Mat gray_image = fp.gray(fp.image);
    //create Gaussian Filter
    vector<vector<double> > filter = fp.createFilter(3, 3, 1);
    Mat gFiltered = fp.useFilter(gray_image, filter); //use Gaussian Filter
    Mat sobel_image = fp.sobel(gray_image);
    Mat nonMax_image = fp.nonMaxSupp(sobel_image);
    //Mat thres = fp.threshold(nonMax_image, 20, 40);
    Mat thres = fp.threshold(nonMax_image, 40, 60);

 
    namedWindow("window1", CV_WINDOW_AUTOSIZE);
    moveWindow("window1", 20, 20); 
    imshow("window1", fp.image);
    waitKey(0);
    imshow("window1", gray_image);
    waitKey(0);
    imshow("window1", sobel_image);
    waitKey(0);
    imshow("window1", nonMax_image);
    waitKey(0);
    imshow("window1", thres);
    waitKey(0);

    int w = thres.cols;
    int h = thres.rows;

    fp.HoughTransform(thres.data, w, h);
    //Search for the line  
    int line_thres = 0;
    if(num==1)
        line_thres = 100;
    else if(num==2)
        line_thres = 130;
    else if(num==3)
        line_thres = 40;
    vector< pair< pair<int, int>, pair<int, int> > > lines = fp.GetLines(line_thres);  
 
    //Draw the results  
    vector< pair< pair<int, int>, pair<int, int> > >::iterator it;  
    for(it=lines.begin(); it!=lines.end(); it++){  
	cout << "point1: " << Point(it->first.first, it->first.second) << " point2:" << Point(it->second.first, it->second.second) << endl; 
	//line(image, Point(0,0), Point(700, 500), Scalar(0, 0, 255), 10, 8);

        line(fp.image, Point(it->first.first, it->first.second), Point(it->second.first, it->second.second), Scalar(0, 0, 255), 5, 8);
    }

    //char c = waitKey(360000);
    //    if(c == '+')
    //        threshold += 5;
    //    if(c == '-')
    //        threshold -= 5;
    //    if(c == 27)
    //        break;
    //Visualize all
  
//    int aw, ah, maxa;  
//    aw = ah = maxa = 0;  
//    const unsigned int* accu = fp.GetAccu(&aw, &ah);  
//   
//    for(int p=0;p<(ah*aw);p++){  
//        if(accu[p] > maxa)  
//            maxa = accu[p];  
//    }  
//    double contrast = 1.0;  
//    double coef = 255.0 / (double)maxa * contrast;  
//   
//    Mat img_accu(ah, aw, CV_8UC3);  
//    for(int p=0;p<(ah*aw);p++){  
//        unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;  
//        img_accu.data[(p*3)+0] = 255;  
//        img_accu.data[(p*3)+1] = 255-c;  
//        img_accu.data[(p*3)+2] = 255-c;  
//    }
//    imshow("window1", img_accu);
    imshow("window1", fp.image);


    waitKey(0);


    return 0;
}
