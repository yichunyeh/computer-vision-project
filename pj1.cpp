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
        Mat gray(Mat); //Grayscale
        vector<vector<double> > createFilter(int, int, double); //Creates a gaussian filter
        Mat useFilter(Mat, vector<vector<double> >); //Use gaussian filter
        Mat sobel(Mat); //Sobel filtering
        Mat nonMaxSupp(Mat); //Non-maxima supp.
	Mat threshold(Mat, int, int); //Double threshold and finalize picture
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

vector< pair< pair<int, int>, pair<int, int> > > find_parallelograms::GetLines(int threshold){  
    vector< pair< pair<int, int>, pair<int, int> > > lines;  
    //vector< int, vector< pair<int, int>, pair<int, int> > paralle;
    if(accu == 0)  
        return lines;  

    for(int r=0; r<accu_h; r++){  
        for(int t=0; t<accu_w; t++){  
            if((int)accu[(r*accu_w) + t] >= threshold){  
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
                int x1, y1, x2, y2;  
                x1 = y1 = x2 = y2 = 0;  

                if(t >= 45 && t <= 135){  
                    ////y = (r - xcos(t))/sin(t)  
                    x1 = 0;  
                    //y1 = ((double)(r-(accu_h/2)) - ((x1-(img_w/2))*cos(t*PI/180.0))) / sin(t*PI/180.0) + (img_h/2);  
		    y1 = ((double)(r-(accu_h/2)) - ((x1-(img_w/2))*cos(t*DEG2RAD))) / sin(t*DEG2RAD) + (img_h/2); 
                    x2 = img_w - 0;  
                    y2 = ((double)(r-(accu_h/2)) - ((x2-(img_w/2))*cos(t*DEG2RAD))) / sin(t*DEG2RAD) + (img_h/2);  
                }  
                else{  
                    //x = (r - y sin(t)) / cos(t);  
                    y1 = 0;  
                    x1 = ((double)(r-(accu_h/2)) - ((y1-(img_h/2))*sin(t*DEG2RAD))) / cos(t*DEG2RAD)+(img_w/2);  
                    y2 = img_h - 0;  
                    x2 = ((double)(r-(accu_h/2)) - ((y2-(img_h/2))*sin(t*DEG2RAD))) / cos(t*DEG2RAD)+(img_w/2);  
                } 
                //slope = (y1-y2) / (x1-x2);                 
		
    
                lines.push_back(pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1,y1), pair<int, int>(x2,y2)));  
   
            }  
        }  
    }  

    cout << "lines: " << lines.size() << " " << threshold << endl;  
    return lines;  
} 
  
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
    Mat image;
    find_parallelograms fp;
    cout << "Using image(1,2 or3):";
    cin >> num;

    if(num==1)
    	image = imread("TestImage1c.jpg", 1);
    else if(num==2)
	 image = imread("TestImage2c.jpg", 1);
    else if(num==3)
	 image = imread("TestImage3.jpg", 1);

    //imwrite("Gray_Image.jpg", gray_image);

    Mat gray_image = fp.gray(image);
    //create Gaussian Filter 
    vector<vector<double> > filter = fp.createFilter(3, 3, 1);
    Mat gFiltered = fp.useFilter(gray_image, filter); //use Gaussian Filter
    Mat sobel_image = fp.sobel(gray_image);
    Mat nonMax_image = fp.nonMaxSupp(sobel_image);
    //Mat thres = fp.threshold(nonMax_image, 20, 40);
    Mat thres = fp.threshold(nonMax_image, 40, 60);

 
    namedWindow("window1", CV_WINDOW_AUTOSIZE);
    moveWindow("window1", 20, 20); 
    imshow("window1", image);
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
        line_thres = 300;
    vector< pair< pair<int, int>, pair<int, int> > > lines = fp.GetLines(line_thres);  
 
    //Draw the results  
    vector< pair< pair<int, int>, pair<int, int> > >::iterator it;  
    for(it=lines.begin(); it!=lines.end(); it++){  
	cout << "point1: " << Point(it->first.first, it->first.second) << " point2:" << Point(it->second.first, it->second.second) << endl; 
	//line(image, Point(0,0), Point(700, 500), Scalar(0, 0, 255), 10, 8);

        line(image, Point(it->first.first, it->first.second), Point(it->second.first, it->second.second), Scalar(0, 0, 255), 5, 8);
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
    imshow("window1", image);


    waitKey(0);


    return 0;
}
