#include<iostream>
#include<string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/calib3d.hpp>
#include<typeinfo>

using namespace cv;
using namespace std;

const string GROUND_TRUTH_FOR_BOARD_IMAGES[][3] = {
	{"DraughtsGame1Move0.JPG", "1,2,3,4,5,6,7,8,9,10,11,12", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move1.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move2.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move3.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move4.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "17,20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move5.JPG", "1,2,3,4,5,7,8,9,10,11,12,22", "20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move6.JPG", "1,2,3,4,5,7,8,9,10,11,12", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move7.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move8.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move9.JPG", "1,2,3,4,5,7,8,10,11,12,22", "20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move10.JPG", "1,2,3,4,5,7,8,10,11,12", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move11.JPG", "1,2,3,4,5,7,8,10,11,16", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move12.JPG", "1,2,3,4,5,7,8,10,11,16", "14,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move13.JPG", "1,2,3,4,5,7,8,11,16,17", "20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move14.JPG", "1,2,3,4,5,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move15.JPG", "1,3,4,5,6,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move16.JPG", "1,3,4,5,6,7,8,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move17.JPG", "1,3,4,5,7,8,9,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move18.JPG", "1,3,4,5,7,8,9,11,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move19.JPG", "1,3,4,5,7,8,9,15,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move20.JPG", "1,3,4,5,8,9,16", "K2,14,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move21.JPG", "1,3,4,5,8,16,18", "K2,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move22.JPG", "1,3,4,5,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move23.JPG", "1,4,5,7,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move24.JPG", "1,4,5,7,8", "K2,11,14,27,28,29,31,32"},
	{"DraughtsGame1Move25.JPG", "1,4,5,8,16", "K2,14,27,28,29,31,32"},
	{"DraughtsGame1Move26.JPG", "1,4,5,8,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move27.JPG", "1,4,5,11,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move28.JPG", "1,4,5,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move29.JPG", "4,5,6,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move30.JPG", "4,5,6,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move31.JPG", "4,5,9,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move32.JPG", "4,5,9,11,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move33.JPG", "4,5,11,14,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move34.JPG", "4,5,11,14,16", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move35.JPG", "4,5,11,16,17", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move36.JPG", "4,5,11,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move37.JPG", "4,5,15,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move38.JPG", "4,5,15,16,17", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move39.JPG", "4,5,15,17,19", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move40.JPG", "4,5,15,17,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move41.JPG", "4,5,17,18,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move42.JPG", "4,5,17,18,19", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move43.JPG", "4,5,17,19,22", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move44.JPG", "4,5,17,19,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move45.JPG", "4,5,19,21,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move46.JPG", "4,5,19,21,22", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move47.JPG", "4,5,19,22,25", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move48.JPG", "4,5,19,25", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move49.JPG", "4,5,19,K30", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move50.JPG", "4,5,19,K30", "K2,20,K26,27,28,29,32"},
	{"DraughtsGame1Move51.JPG", "4,5,19,K23", "K2,20,27,28,29,32"},
	{"DraughtsGame1Move52.JPG", "4,5,19", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move53.JPG", "4,5,23", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move54.JPG", "4,5,23", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move55.JPG", "4,5,26", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move56.JPG", "4,5,26", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move57.JPG", "4,5,K31", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move58.JPG", "4,5,K31", "K2,11,20,27,28,29"},
	{"DraughtsGame1Move59.JPG", "4,5,K24", "K2,11,20,28,29"},
	{"DraughtsGame1Move60.JPG", "4,5", "K2,11,19,20,29"},
	{"DraughtsGame1Move61.JPG", "4,9", "K2,11,19,20,29"},
	{"DraughtsGame1Move62.JPG", "4,9", "K2,11,19,20,25"},
    {"DraughtsGame1Move63.JPG", "4,14", "K2,11,19,20,25"},
    {"DraughtsGame1Move64.JPG", "4,14", "K2,11,19,20,22"},
    {"DraughtsGame1Move65.JPG", "4,18", "K2,11,19,20,22"},
    {"DraughtsGame1Move66.JPG", "4", "K2,11,15,19,20"},
    {"DraughtsGame1Move67.JPG", "8", "K2,11,15,19,20"},
    {"DraughtsGame1Move68.JPG", "", "K2,K4,15,19,20"}
};

void assignPart1();
void calcHist();
void assignPart2(string path,int itr_no);
int check_piece(Point center);
void compute_matrix();
bool isEqual(Mat first, Mat second);
void findFrames();
void assignPart3();
void lineDetection();
void usingContours();
void chessboardcorners();
void assignPart5();
int FindHSVVals();
void compute_ext_matrix(int white_king_count,int black_king_count);
string findData(vector<string> current, vector<string> previous);

Mat imgWrap, matrix, backgroundG, backgroundW, backgroundBP, backgroundWP, maskW, maskG, image, imgHSV, mask, remove_bp_low, remove_bp_high;
Mat crop;

Mat maskEmpty, mask_bp, mask_wp;
float w = 300, h = 300;

string empty_path = "DraughtsGame1EmptyBoard.JPG";

Mat empty_img = imread(empty_path);

int hmin = 63, smin = 50, vmin = 51;
int hmax = 150, smax = 135, vmax = 164;

// Old black square vals
// Scalar greensquareLow = Scalar(54, 40, 51);
// Scalar greensquareHigh = Scalar(170, 135, 127);

Scalar greensquareLow = Scalar(63, 50, 51);
Scalar greensquareHigh = Scalar(150, 135, 164);

// Old white square vals
// Scalar whitesquareLow = Scalar(0, 80, 150);
// Scalar whitesquareHigh = Scalar(150, 170, 255);
Scalar whitesquareLow = Scalar(0, 58, 126);
Scalar whitesquareHigh = Scalar(15, 161, 239);
// int hminG = 20, sminG = 5, vminG = 36;
// int hmaxG = 255, smaxG = 255, vmaxG = 255;

Scalar whitepieceLow = Scalar(0, 0, 225);
Scalar whitepieceHigh = Scalar(179, 255, 255);

Scalar blackpieceLow = Scalar(0, 0, 0);
Scalar blackpieceHigh = Scalar(16, 255, 129);

vector<Vec3f> circles;

vector<string> pdn_parser(string str)
{   
    vector<string> data;
    stringstream ss(str);
    while(ss.good())
    {
        string substr;
        getline(ss, substr, ',');
        stringstream ss2(substr);
        while(ss2.good())
        {
            getline(ss2, substr, '"');
            stringstream ss3(substr);
            while(ss3.good())
            {
                getline(ss3, substr, 'K');
                stringstream ss4(substr);
                while(ss4.good())
                {
                    getline(ss4, substr, ' ');
                }
            }
        }
        data.push_back(substr);
    }
    return data;

}
vector<string> part5_parser(string str)
{   
    vector<string> data;
    stringstream ss(str);
    while(ss.good())
    {
        string substr;
        getline(ss, substr, ',');
        stringstream ss2(substr);
        while(ss2.good())
        {
            getline(ss2, substr, '"');
            stringstream ss3(substr);
            while(ss3.good())
            {
                getline(ss3, substr, 'K');
                
            }
        }
        data.push_back(substr);
    }
    return data;

}

// We are going to create a structure to keep track of the black squares numbers and use this data to deteermine the ground truth data and build the confusion matrix
struct{
    Point tl;
    Point br;
    string sq_no;

}PDN_Data[32];

// Creating the detected pdn variable for the pieces data
string detected[69][3];

Mat test;

void wrapImage(Mat img){
    Point2f src[4] = { {114, 17}, {355, 20}, {53, 245}, {433, 241} };
    Point2f dst[4] = { {0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h} };
    matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWrap, matrix, Point(w, h));
}

void getContours(Mat imgBinary, Mat img, string category)
{
    string numbers[32] = {"12", "4", "28", "20", "32", "16", "8", "24", "11", "3", "27", "19", "15", "31", "23", "7", "10", "18", "2", "26", "14", "22", "6", "30", "9", "17", "1", "25", "29", "21", "13", "5"};
    vector<vector<Point>> contours;
	Point center;
    vector<Vec4i> hierarchy;
    int squares = 0;
    findContours(imgBinary, contours, RETR_TREE, CHAIN_APPROX_NONE);
    for ( int i = 0; i < contours.size(); i++ )
    {
        // For fine-tuning the image we are going to bound the counters to rectangles		
		Moments M = moments(contours[i]);
		Point center(M.m10/M.m00, M.m01/M.m00);
		int inContour;
		if (contourArea(contours[i]) > 500)
		{
        Rect boundRect = boundingRect(contours[i]);
			if(boundRect.area()>500 && (boundRect.width < 70 || boundRect.height < 70))
			{
                // This part is to detect whether there is a piece inside the rectangle.. If so then please do not detect this as an empty square
                // The data inside circles is globally created every time the program is run using HoughCircles
                int struct_no = 0;
				for (int j = 0; j < circles.size(); j++)
				{
					Vec3i c = circles[j];
					Point center = Point(c[0],c[1]);
					inContour = pointPolygonTest(contours[i], center, false);
					if (inContour == 1 || inContour == 0){goto noprint;}
				}
					rectangle(img, boundRect.tl(), boundRect.br(), Scalar(255 ,0 ,0), 3);
					rectangle(maskEmpty, boundRect.tl(), boundRect.br(), Scalar(255 ,0 ,0), FILLED);
					if (category == "W_sq")
					putText(img, "W", (center), FONT_HERSHEY_COMPLEX, 0.6,Scalar(20, 208, 14), 1);
					else
                    {   
                        if (category == "Empty")
                        {
					        putText(img, numbers[squares], (center), FONT_HERSHEY_COMPLEX, 0.6,Scalar(0, 0, 240), 1);
                            PDN_Data[stoi(numbers[squares])-1].tl = boundRect.tl();
                            PDN_Data[stoi(numbers[squares])-1].br = boundRect.br();
                            PDN_Data[stoi(numbers[squares])-1].sq_no = numbers[squares];
                            struct_no++;
                        }
                        else
                            putText(img, "B", (center), FONT_HERSHEY_COMPLEX, 0.6,Scalar(0, 0, 240), 1);
				    }
				noprint:;
			}
            squares++;
		}
    }

}

void getPieces(Mat imgBinary, Mat img, string category, string path, int itr_no){
	vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    int index;
	findContours(imgBinary, contours, RETR_TREE, CHAIN_APPROX_NONE);
	vector<vector<Point> > contours_poly( contours.size() );
    vector<Point2f>centers( contours.size() );
    vector<float>radius( contours.size() );
	for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
    }

	for ( int i = 0; i < contours.size(); i++ ){
		Moments M = moments(contours[i]);
		Point center(M.m10/M.m00, M.m01/M.m00);
		if (contourArea(contours[i]) > 310){
			if (category == "W")
			{
				circle(img, centers[i], (int)radius[i], Scalar(255, 0, 255), 2 );
				putText(img, "Wp", (center), FONT_HERSHEY_COMPLEX, 0.4,Scalar(0, 0, 0), 1);
                index = check_piece(center);
                if (index != -1){
                    detected[itr_no][0] = path;
                    detected[itr_no][1] += to_string(index+1);
                    detected[itr_no][1] += ",";
                }
			}
			else
			{
				circle(img, centers[i], (int)radius[i], Scalar(255, 253, 0), 2 );
				putText(img, "Bp", (center), FONT_HERSHEY_COMPLEX, 0.4,Scalar(255, 255, 255), 1);
                index = check_piece(center);
                if (index != -1)
                {
                    detected[itr_no][0] = path;
                    detected[itr_no][2] += to_string(index+1);
                    detected[itr_no][2] += ",";
                }

			}
		}
	}
	

}

int check_piece(Point center){
    for( int i = 0 ; i < 32 ; i++ )
    {
        if( int(PDN_Data[i].tl.x) < int(center.x) && int(center.x) < int(PDN_Data[i].br.x) && int(PDN_Data[i].br.y) > int(center.y) && int(center.y) > int(PDN_Data[i].tl.y) ){
            return i;  
        }
    }
    return -1;
}

void detectCircles(Mat img){

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	

	medianBlur(gray, gray, 5);
	HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
	gray.rows/100, // Change this value to detect circles with different distances to each other
	100, 15, 11, 15 // Change the last two values 
	// (min_radius and max_radius) to detect larger circles
	);
    // cout<<circles.size()<<endl;

	for (size_t i = 0; i < circles.size(); i++){
		Vec3i c = circles[i];
		Point center = Point(c[0],c[1]);
		// circle the piece
		// cout<<img[c[0], c[1]]; 
		int radius = c[2];
		// circle(img, center, radius, Scalar(255, 0, 255), FILLED);

	}
}

int main(){
    // FindHSVVals();
    cout<<"Assignment part 1"<<endl;
    assignPart1();
    cout<<"Part 2 - Confusion Matrix"<<endl;
	
    wrapImage(empty_img);
    imgWrap.copyTo(backgroundG);

    // This call is for storing the data for the empty board
    cvtColor(imgWrap, imgWrap, COLOR_BGR2HSV);

    inRange(imgWrap, greensquareLow, greensquareHigh, maskG);
    getContours(maskG, backgroundG, "Empty");
    imshow("Empty Board", backgroundG);
    
    // Calling the part 2 of the assignment ------------

    for( int i = 0 ; i < 69 ; i++ )
    {
        string path = "DraughtsGames1Moves/DraughtsGame1Move";
        path = path + to_string(i);
        path = path + ".jpg";
        // cout<<path<<endl;
        assignPart2(path, i);
    }
    compute_matrix();

    // Part 2 done -----------
    cout<<"Part 3 "<<endl;
    // Part 3 starts here --------------------

    assignPart3();
    cout<<endl;

    // Assignment part 4 ---------------------
    lineDetection();
    usingContours();
    // chessboardcorners();

    cout<<"Part 5 extended confusion matrix"<<endl;

    // Assignment part 5
    assignPart5();
    cout<<endl;
    // cout<< part3<< endl; 

    waitKey(0);
    return 0;
}

// int main(){
    
//     chessboardcorners();

//     waitKey(0);
//     return 0;
// }

void assignPart5()
{
    for( int i = 0 ; i < 69 ; i++ )
    {
        string path = "DraughtsGames1Moves/DraughtsGame1Move";
        path = path + to_string(i);
        path = path + ".jpg";
        // cout<<path<<endl;
        assignPart2(path, i);
    }
    int white_king_count = 0;
    int black_king_count = 0;

    for (int i = 0; i < 69 ; i++)
    {
        vector<string> data = pdn_parser(detected[i][1]);
        vector<string> data2 = pdn_parser(detected[i][2]);

        for(int k = 0; k < data.size(); k++)
        {
            if(data[k] == "29" || data[k] == "30" || data[k] == "31" || data[k] == "32")
            {
                data[k] = "K" + data[k];
                white_king_count++;
            }
        }
        for(int k = 0; k < data2.size(); k++)
        {
            if(data2[k] == "1" || data2[k] == "2" || data2[k] == "3" || data2[k] == "4")
            {
                data2[k] = "K" + data2[k];
                black_king_count++;
            }
        }
        
    }
    // After all the kings are detected let's calculate the extended confusion matrix
    compute_ext_matrix(white_king_count, black_king_count);
}

void compute_ext_matrix(int white_king_count, int black_king_count)
{
    vector<string> data_white;
    vector<string> data_black;
    vector<string> predicted_white;
    vector<string> predicted_black;

    vector<vector<int> > confusion_matrix(5,vector<int>(5, 0));
    for (size_t i = 0; i < 69; i++)
    {
        data_white = part5_parser(GROUND_TRUTH_FOR_BOARD_IMAGES[i][1]);
        data_black = part5_parser(GROUND_TRUTH_FOR_BOARD_IMAGES[i][2]);
        predicted_white = pdn_parser(detected[i][1]);
        predicted_black = pdn_parser(detected[i][2]);
        
        for(int k = 0; k < predicted_white.size(); k++)
        {
            if(predicted_white[k] == "29" || predicted_white[k] == "30" || predicted_white[k] == "31" || predicted_white[k] == "32")
            {
                predicted_white[k] = "K" + predicted_white[k];
            }
        }
        for(int k = 0; k < predicted_black.size(); k++)
        {
            if(predicted_black[k] == "1" || predicted_black[k] == "2" || predicted_black[k] == "3" || predicted_black[k] == "4")
            {
                predicted_black[k] = "K" + predicted_black[k];
            }
        }

        // comparing data for white pieces to fill the confusion matrix
        for( int j = 0; j < data_white.size() ; j++)
        {
            for( int k = 0; k < predicted_white.size() ; k++)
            {
                // ideal case match and found
                if (data_white[j] == predicted_white[k])
                {   
                    if (data_white[j] != "-1")
                        confusion_matrix[1][1] += 1;
                    data_white[j] = "-1";
                    predicted_white[k] = "-1";
                }
            }
                
        }
        // Checking if white piece was present but not detected and if a white piece was detected but wasn"t present
        for (int j = 0 ; j < data_white.size(); j++)
        {   
            
            if (data_white[j] != "-1" && data_white[j] != "")
            {
                confusion_matrix[0][1] += 1;
            }
        }
        for (int j = 0 ; j < predicted_white.size(); j++)
        {
            if (predicted_white[j] != "-1" && predicted_white[j] != "")
            {   
                // cout<<"Thats the one"<<predicted_white[j]<<"Thanks";
                confusion_matrix[1][0] += 1;
                // cout<<i<<"-"<<j<<"\t"<<predicted_white[j]<<endl;
            }
        }

        // comparing data for black pieces to fill the confusion matrix
        for( int j = 0; j < data_black.size() ; j++)
        {
            for(int k = 0; k < predicted_black.size(); k++)
            {
                // ideal case match and found
                if (data_black[j] == predicted_black[k])
                {   
                    if (data_black[j] != "-1")
                        confusion_matrix[2][2] += 1;
                    
                    data_black[j] = "-1";
                    predicted_black[k] = "-1";
                }
            }
                
        }

        // Checking if black piece was present but not detected and if a black piece was detected but wasn't present
        for (int j = 0 ; j < data_black.size(); j++)
        {
            
            if (data_black[j] != "-1" && data_black[j] != "")
            {
                confusion_matrix[0][2] += 1;
            }
        }
        for (int j = 0 ; j < predicted_black.size(); j++)
        {
            if (predicted_black[j] != "-1" && predicted_black[j] != "")
            {
                confusion_matrix[2][0] += 1;
            }
                
        }

        // Logic to detect for no pice - no piece detection
        // Black pieces 
        confusion_matrix[0][0] += 16 - data_black.size();

        // white pieces 
        confusion_matrix[0][0] += 16 - data_white.size();

    }

    // White king counter
    confusion_matrix[1][1] -= white_king_count;
    confusion_matrix[2][2] -= black_king_count;
    confusion_matrix[3][3] = white_king_count;
    confusion_matrix[4][4] = black_king_count;



    for (int i = 0; i < 5; i++)
    {
        for (int j = 0 ; j < 5 ; j++)
            cout<<"\t"<<confusion_matrix[i][j];
    cout<<endl;
    }
}


void chessboardcorners()
{
    Mat gray;
    Size patternsize(8,6); //interior number of corners
    Mat chesscorners = imread("DraughtsGames1Moves/DraughtsGame1Move0.JPG"); //source image
    cvtColor(chesscorners,gray, COLOR_BGR2GRAY);
    vector<Point2f> corners; //this will be filled by the detected corners
    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);
    if(patternfound)
    // cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
    //     TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    drawChessboardCorners(chesscorners, patternsize, Mat(corners), patternfound);
    imshow("chessboard corners", chesscorners);
}

void usingContours()
{
    Mat conImg = imread("DraughtsGames1Moves/DraughtsGame1Move0.JPG");
    // Convert to gray-scale
    Mat gray;
    cvtColor(conImg, gray, COLOR_BGR2GRAY);
    // Store the edges 
    Mat edges, blur;
    // Find the edges in the image using canny detector
    GaussianBlur(gray, blur, Size(3,3), 0);
    // Canny(blur, edges, 20, 150);
    Canny(blur, edges, 50, 200);
    Mat kernel = Mat::ones(2, 2, CV_8UC1);
    dilate(edges,edges,kernel);
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( edges, contours, hierarchy,RETR_EXTERNAL, CHAIN_APPROX_NONE );

    for (int contour_number=0; (contour_number<contours.size()); contour_number++)
    {
        if(contourArea(contours[contour_number])> 1270 )
        {
            // cout<<contourArea(contours[contour_number]);
            Scalar colour(255,0,0);
            drawContours( conImg, contours, contour_number,colour, 2, 8, hierarchy );
        }
    
    }
    imshow("using contours", conImg);
    imwrite("part4_Contours.png", conImg);
    // imshow("canny image", edges);
}

void lineDetection()
{
    // Read the image as gray-scale
    Mat lineImg = imread("DraughtsGames1Moves/DraughtsGame1Move0.JPG");
    // Convert to gray-scale
    Mat gray;
    cvtColor(lineImg, gray, COLOR_BGR2GRAY);
    // Store the edges 
    Mat edges, blur;
    // Find the edges in the image using canny detector
    GaussianBlur(gray, blur, Size(5,5), 0);
    // Canny(blur, edges, 20, 150);
    Canny(blur, edges, 20, 150);
    // Create a vector to store lines of the image
    vector<Vec4i> lines;
    // Apply Hough Transform
    HoughLinesP(edges, lines,1, CV_PI/180, 70, 1, 20);
    // Draw lines on the image

    // filtering with rough values just the top line of the board
    Point tl,tr,bl,br;
    for (size_t i=0; i<lines.size(); i++) {
        Vec4i l = lines[i];
        if( (l[0] < 200 && l[0] > 50 && l[1] > 5 && l[1] < 50) && ( l[2] > 355 && l[3] > 10))
            {
                // line(lineImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 2, LINE_AA);
                string text = to_string(l[0]);
                text += ",";
                text += to_string(l[1]);
                putText(lineImg, text, Point(l[0]+10, l[1]+20), FONT_HERSHEY_COMPLEX, 0.4,Scalar(0,0,255), 1);
                text = to_string(l[2]);
                text += ",";
                text += to_string(l[3]);
                putText(lineImg, text, Point(l[2]-60, l[3]+20), FONT_HERSHEY_COMPLEX, 0.4,Scalar(0,0,255), 1);
                tl = Point(l[0], l[1]);
                tr = Point(l[2], l[3]);
            }
    }

    for (size_t i=0; i<lines.size(); i++) {
        Vec4i l = lines[i];
        if( (l[0] < 100 && l[0] > 30 && l[1] > 215 && l[1] < 251) && ( l[2] > 355 && l[3] > 10))
            {
                // line(lineImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 2, LINE_AA);
                string text = to_string(l[0]);
                text += ",";
                text += to_string(l[1]);
                putText(lineImg, text, Point(l[0]+20, l[1]-20), FONT_HERSHEY_COMPLEX, 0.4,Scalar(0,0,255), 1);
                text = to_string(l[2]);
                text += ",";
                text += to_string(l[3]);
                putText(lineImg, text, Point(l[2]-80, l[3]-20), FONT_HERSHEY_COMPLEX, 0.4,Scalar(0,0,255), 1);
                bl = Point(l[0], l[1]);
                br = Point(l[2], l[3]);
            }
    }
    line(lineImg, tl, tr, Scalar(255,0,0), 2, LINE_AA);
    line(lineImg, tl, bl, Scalar(255,0,0), 2, LINE_AA);
    line(lineImg, br, tr, Scalar(255,0,0), 2, LINE_AA);
    line(lineImg, br, bl, Scalar(255,0,0), 2, LINE_AA);

    // Show result image
    imshow("Result Image", lineImg);
    imwrite("part4_Line.png", lineImg);
    // imshow("edges", edges);
}

void assignPart3()
{
    findFrames();
    string part3;
    for( int i = 0 ; i < 69 ; i++ )
    {
        string path = "part3/DraughtsGame1Move";
        path = path + to_string(i);
        path = path + ".jpg";
        // cout<<path<<endl;
        assignPart2(path, i);
    }

    for( int i = 0 ; i < 69 ; i++ )
    {
        if(i != 0)
            part3+= " ";
        string result_white, result_black;
        vector<string> white_data = pdn_parser(detected[i][1]);
        vector<string> black_data = pdn_parser(detected[i][2]);
        if(i==0)
            continue;
        vector<string> previous_white = pdn_parser(detected[i-1][1]);
        vector<string> previous_black = pdn_parser(detected[i-1][2]);
        // if(i%2 == 1)
        result_white = findData(white_data, previous_white);
        if (result_white != "")
        {   
            part3 += "W:";
            part3 += result_white;
        }
        part3 += " ";
        result_black = findData(black_data, previous_black);
        if (result_black != "")
        {
            part3 += "B:";
            part3 += result_black;
        }
             
    }
    cout<<part3<<endl;
}

string findData(vector<string> current, vector<string> previous)
{
    string first,second, result;
    int i, j;
    for( i = 0 ; i < current.size(); i++)
    {
        int found = 0;
        for( j = 0 ; j < previous.size() ; j++)
        {
            if(current[i] == previous[j])
                {found = 1;}
        }
        if (found == 0)
        {
            second = current[i];
        }
    }
    for( i = 0 ; i < previous.size(); i++)
    {
        int found = 0;
        for( j = 0 ; j < current.size() ; j++)
        {
            if(previous[i] == current[j])
                {found = 1;}
        }
        if (found == 0)
        {
            first = previous[i];
        }
    }
    result = first + "-";
    result += second;
    if(first !="" && second !="")
        return result;
    else
        return "";
}

void findFrames()
{
    string video = "DraughtsGame1.avi";
    VideoCapture cap(video);
    Mat frame, videoMask, difference;
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
    unsigned long counter = 0;
    int saveCount = 21;
    int framecount = 0;
    int flag = 0;
    Mat previous;
    while(cap.read(frame))
    {
        string framename = "part3/DraughtsGame1Move";
        if (counter == 0)
        {
            frame.copyTo(videoMask);
            counter++;
            continue;
        }
        pBackSub->apply(frame, videoMask);
        saveCount++;
        int TotalNumberOfPixels = videoMask.rows * videoMask.cols;
        int ZeroPixels = TotalNumberOfPixels - countNonZero(videoMask);
        if (ZeroPixels > 126670)
        {   
            if (framecount != 0)
            {
                Mat scoreImg,framewrap, previouswrap;
                double maxScore;
                Point2f source[4] = { {114, 17}, {355, 20}, {53, 245}, {433, 241} };
                Point2f destination[4] = { {0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h} };
                matrix = getPerspectiveTransform(source, destination);
                warpPerspective(previous, previouswrap, matrix, Point(w, h));
                warpPerspective(frame, framewrap, matrix, Point(w, h));

                matchTemplate(previouswrap, framewrap, scoreImg, TM_CCOEFF_NORMED);
                minMaxLoc(scoreImg, 0, &maxScore);
                if(maxScore > 0.993)
                    continue;
            }
            framename += to_string(framecount);
            framename += ".jpg";
            framecount++;
            saveCount = 0;
            frame.copyTo(previous);
            imshow("Frame", frame);
            imwrite(framename, frame);
        }
        int keyboard = waitKey(10);
        if (keyboard == 'q' || keyboard == 27)
            break;
        counter++;
    }    
}

// Function to check if two images are identical
bool isEqual(Mat first, Mat second)
{
    Mat dst;
    vector<Mat>channels;
    int count = 0;
    bitwise_xor(first, second, dst);
    split(dst, channels);
    for (int ch = 0; ch<dst.channels();ch++){
        count += countNonZero(channels[ch]);
    }
    return count == 0 ? true : false;

}

// Function to compute the comfusion matrix after extracting all the predicted data
void compute_matrix()
{
    vector<string> data_white;
    vector<string> data_black;
    vector<string> predicted_white;
    vector<string> predicted_black;

    vector<vector<int> > confusion_matrix(3,vector<int>(3, 0));
    for (size_t i = 0; i < 69; i++)
    {
        data_white = pdn_parser(GROUND_TRUTH_FOR_BOARD_IMAGES[i][1]);
        data_black = pdn_parser(GROUND_TRUTH_FOR_BOARD_IMAGES[i][2]);
        predicted_white = pdn_parser(detected[i][1]);
        predicted_black = pdn_parser(detected[i][2]);

        // comparing data for white pieces to fill the confusion matrix
        for( int j = 0; j < data_white.size() ; j++)
        {
            for( int k = 0; k < predicted_white.size() ; k++)
            {
                // ideal case match and found
                if (data_white[j] == predicted_white[k])
                {   
                    if (data_white[j] != "-1")
                        confusion_matrix[1][1] += 1;
                    data_white[j] = "-1";
                    predicted_white[k] = "-1";
                }
            }
                
        }
        // Checking if white piece was present but not detected and if a white piece was detected but wasn"t present
        for (int j = 0 ; j < data_white.size(); j++)
        {   
            
            if (data_white[j] != "-1" && data_white[j] != "")
            {
                confusion_matrix[0][1] += 1;
            }
        }
        for (int j = 0 ; j < predicted_white.size(); j++)
        {
            if (predicted_white[j] != "-1" && predicted_white[j] != "")
            {   
                cout<<"Thats the one"<<predicted_white[j]<<"Thanks";
                confusion_matrix[1][0] += 1;
                // cout<<i<<"-"<<j<<"\t"<<predicted_white[j]<<endl;
            }
        }

        // comparing data for black pieces to fill the confusion matrix
        for( int j = 0; j < data_black.size() ; j++)
        {
            for(int k = 0; k < predicted_black.size(); k++)
            {
                // ideal case match and found
                if (data_black[j] == predicted_black[k])
                {   
                    if (data_black[j] != "-1")
                        confusion_matrix[2][2] += 1;
                    data_black[j] = "-1";
                    predicted_black[k] = "-1";
                }
            }
                
        }

        // Checking if black piece was present but not detected and if a black piece was detected but wasn't present
        for (int j = 0 ; j < data_black.size(); j++)
        {
            
            if (data_black[j] != "-1" && data_black[j] != "")
            {
                confusion_matrix[0][2] += 1;
            }
        }
        for (int j = 0 ; j < predicted_black.size(); j++)
        {
            if (predicted_black[j] != "-1" && predicted_black[j] != "")
            {
                confusion_matrix[2][0] += 1;
            }
                
        }

        // Logic to detect for no pice - no piece detection
        // Black pieces 
        confusion_matrix[0][0] += 16 - data_black.size();

        // white pieces 
        confusion_matrix[0][0] += 16 - data_white.size();

    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0 ; j < 3 ; j++)
            cout<<"\t"<<confusion_matrix[i][j];
    cout<<endl;
    }
}





// // Code To Find HSV Values

int FindHSVVals()
{
    // Code to run
    cout << "OpenCV is Here World" << endl;
    
    Mat image = imread("DraughtsGame1EmptyBoard.JPG",1);
    Mat imgHSV, mask;
    cvtColor(image, imgHSV, COLOR_BGR2HSV);
    // namedWindow("Trackbars", (640,200));
	namedWindow("Trackbars");
    createTrackbar("Hue Min", "Trackbars", &hmin, 179);
    createTrackbar("Hue Max", "Trackbars", &hmax, 179);
    createTrackbar("Sat Min", "Trackbars", &smin, 255);
    createTrackbar("Sat Max", "Trackbars", &smax, 255);
    createTrackbar("Val Min", "Trackbars", &vmin, 255);
    createTrackbar("Val Max", "Trackbars", &vmax, 255);


    while(true){
    
        Scalar lower(hmin, smin, vmin);
        Scalar upper(hmax, smax, vmax);
        inRange(imgHSV, lower, upper, mask);
        imshow("Find HSV Vals for Part 2",mask);
        waitKey(1);
        
    }
return 0;
}


void assignPart2(string path, int itr_no){
    Mat detectPiece;
    Mat img = imread(path);
    wrapImage(img);
    
	imgWrap.copyTo(detectPiece);
    
    imgWrap.copyTo(backgroundG);
	imgWrap.copyTo(backgroundW);
    imgWrap.copyTo(backgroundBP);
    imgWrap.copyTo(backgroundWP);


	imgWrap.copyTo(maskEmpty);

	// Showing the regions for green spaces

	// Detecting circles for pices and coloring with black
	// if (path != "DraughtsGame1EmptyBoard.JPG")
	detectCircles(imgWrap);
	

    cvtColor(imgWrap, imgWrap, COLOR_BGR2HSV);

    inRange(imgWrap, greensquareLow, greensquareHigh, maskG);
    getContours(maskG, backgroundG, "G_sq");
    



	// Showing the regions for white spaces 
	
	inRange(imgWrap, whitesquareLow, whitesquareHigh, maskW);
    getContours(maskW, backgroundW, "W_sq");
    

	// Showing the regions for White Pieces
    if(circles.size() != 0){

    
        inRange(imgWrap, whitepieceLow, whitepieceHigh, mask_wp);
        getPieces(mask_wp, backgroundWP, "W", path, itr_no);
        
        // detected[itr_no][1][ detected[itr_no][1].size() - 1 ] = '\"';

        inRange(imgWrap, blackpieceLow, blackpieceHigh, mask_bp);
        getPieces(mask_bp, backgroundBP, "B", path, itr_no);
        // detected[itr_no][2][ detected[itr_no][2].size() - 1 ] = '\"';
        

    }
}

// ---------------------------------------------------------------- THIS IS THE PART 1 OF THE ASSIGNMENT ----------------------------------------------------------------

int calcHist(Mat predicted)
{

    Mat groundtruth = imread("GT.png"); // ground truth image provided by the professor for part 1
    Mat hsv_groundtruth, hsv_predicted;
    cvtColor( groundtruth, hsv_groundtruth, COLOR_BGR2HSV );
    cvtColor( predicted, hsv_predicted, COLOR_BGR2HSV );
    Mat hsv_half_down = hsv_groundtruth( Range( hsv_groundtruth.rows/2, hsv_groundtruth.rows ), Range( 0, hsv_groundtruth.cols ) );
    int h_bins = 50, s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    // Use the 0-th and 1-st channels
    int channels[] = { 0, 1 };
    const char* method[4] = { "Correlation", "Chi-Square", "Intersection", "Bhattacharyya Distance" };
    Mat hist_base, hist_half_down, hist_test1, hist_test2;
    calcHist( &hsv_groundtruth, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
    normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
    calcHist( &hsv_predicted, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
    normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );
    for( int compare_method = 0; compare_method < 4; compare_method++ )
    {
        double base_base = compareHist( hist_base, hist_base, compare_method );
        double base_test1 = compareHist( hist_base, hist_test1, compare_method );
        cout << method[compare_method]<<"-->" << " Provided_Ground_Truth, My_Prediction : "
             <<  base_base<< " / " << base_test1 <<endl;
    }
    cout << "Done \n";
    return 0;
}


void assignPart1(){
    Mat imgGrey, mask, imgFinal, imgMask, imgHSV;
    float morph_size = 1;

    // Starting with white pieces 
    Mat img = imread("DraughtsGames1Moves/DraughtsGame1Move0.JPG");
    img.copyTo(imgFinal);
    img.copyTo(imgMask);

    cvtColor(img, imgGrey, COLOR_BGR2GRAY);
    threshold(imgGrey, mask, 187, 255, THRESH_BINARY);
    Mat element = getStructuringElement(
        MORPH_RECT,
        Size(2 * morph_size + 1,
             2 * morph_size + 1),
        Point(morph_size, morph_size));
    morphologyEx(mask, mask, MORPH_OPEN, element);
    cvtColor(mask, mask, COLOR_GRAY2BGR);
    imgFinal.setTo(Scalar(255, 255, 255), mask);
    imgMask.setTo(Scalar(255, 255, 255), mask);

    
    // Moving onto green squares
    cvtColor(img, imgGrey, COLOR_BGR2GRAY);
    threshold(imgGrey, mask, 80, 255, THRESH_BINARY_INV);
    morph_size = 1;
    element = getStructuringElement(
        MORPH_RECT,
        Size(2 * morph_size + 1,
             2 * morph_size + 1),
        Point(morph_size, morph_size));
    morphologyEx(mask, mask, MORPH_ELLIPSE, element);
    cvtColor(mask, mask, COLOR_GRAY2BGR);
    imgFinal.setTo(Scalar(0, 0, 255), mask);
    imgMask.setTo(Scalar(255, 255, 255), mask);
    imshow("YAYYY", imgFinal);

    // Moving onto black Pieces
    cvtColor(imgMask, imgGrey, COLOR_BGR2GRAY);
    threshold(imgGrey, mask, 100, 255, THRESH_BINARY_INV);
    morph_size = 1;
    element = getStructuringElement(
        MORPH_RECT,
        Size(2 * morph_size + 1,
             2 * morph_size + 1),
        Point(morph_size, morph_size));
    morphologyEx(mask, mask, MORPH_CLOSE, element);
    cvtColor(mask, mask, COLOR_GRAY2BGR);
    imgFinal.setTo(Scalar(0, 0, 0), mask);
    imgMask.setTo(Scalar(255, 255, 255), mask);

    // Moving onto white squares 
    
    cvtColor(imgMask, imgHSV, COLOR_BGR2HSV);
	inRange(imgHSV, Scalar(10, 102, 130), Scalar(15, 135, 243), mask);
    imshow("new mask for white squares", mask);
    morph_size = 1;
    element = getStructuringElement(
        MORPH_RECT,
        Size(2 * morph_size + 1,
             2 * morph_size + 1),
        Point(morph_size, morph_size));
    morphologyEx(mask, mask, MORPH_CLOSE, element);
    cvtColor(mask, mask, COLOR_GRAY2BGR);
    imgFinal.setTo(Scalar(0, 255, 0), mask);
    imgMask.setTo(Scalar(255, 255, 255), mask);

    // Moving onto the region that is not part of the board
    cvtColor(imgMask, imgGrey, COLOR_BGR2GRAY);
    threshold(imgGrey, mask, 154, 255, THRESH_BINARY_INV);
    imshow("BINARY", mask);
    morph_size = 1;
    element = getStructuringElement(
        MORPH_RECT,
        Size(2 * morph_size + 1,
             2 * morph_size + 1),
        Point(morph_size, morph_size));
    morphologyEx(mask, mask, MORPH_CLOSE, element);
    cvtColor(mask, mask, COLOR_GRAY2BGR);
    imgFinal.setTo(Scalar(255, 0, 0), mask);
    imshow("Part 1 Final Image", imgFinal);
    imshow("Mask for table", mask);

    imwrite("IMGFINAL_BINARY.png", imgFinal);
    calcHist(imgFinal);
}