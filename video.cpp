#include"opencv2/opencv.hpp"  
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include"opencv2/features2d/features2d.hpp"
using namespace cv;  
#include <iostream> 
#include<cmath>
#include<time.h>
#include<stdlib.h>
//#include <opencv2\Blob_detection.hpp>
using namespace std;  
//运动物体检测函数声明  
#define M_PI 3.14
//Mat MoveDetect(Mat temp, Mat frame);  


class VideoBackground
{
//private:
	/*typedef struct ForeBackground
	{
		Mat fore;
		Mat back;
	}ForeBackground;*/
private:
	//string filename;
	int k;           //单高斯模型的个数k，偏差阈值D
	int width,height;
	double D;
	double alpha;       //学习速率
	double wightThresh ;
	double inSigma;
	double bgTol;    
	double ***muBlue;   //三维数组指针 蓝色通道的均值
	double ***muGreen;
	double ***muRed;
	double ***w;        //初始化权重矩阵
	double ***sigma;    //标准差矩阵
	Mat fore;
	Mat back;
	//Mat frame;
	//struct 

public:
	//VideoBackground();
	VideoBackground(string file);    //构造函数初始化用于聚类的

	//void saveImage() ;             //保存随机一部分帧 用于背景计算
	Mat gaussBackground(Mat img);           //计算背景  混合高斯模型
	Mat gaussBackground(string videoname);
	void updateBackground(int i,int j,Mat frame,int match);          //更新背景
	bool absoluteDistance(int xVal, double mu, double sigma);//计算绝对距离
	void updateWeights(int i ,int j,int l);                //跟新权重
	void updateMeans(int i ,int j,int l,int blue,int green,int red,double rho);  //更新均值
	void updateSigma(int i,int j,int l,int blue,int green,int red,double rho);   //更新标准差
	void sort(int i,int j);          //排序
	bool matePixel(int i,int j,int m,double Val,Mat frame);
	void updateForeground(int i,int j);
	void initialize();
	
	//void createAlphaMat(Mat &mat);
};
void saveImage(string file);
int main()  
{  

//	VideoCapture video("1.wmv");
//	if(!video.isOpened()){
//		return -1;
//	}
//	//VideoBackground video1("2.wmv");
//   // VideoCapture video(0);//定义VideoCapture类video  
//   // if (!video.isOpened())  //对video进行异常检测  
//   // {  
//    //    cout << "video open error!" << endl;  
//    //    return 0;  
//   // }  
//    while(1)
//{
//    int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);//获取帧数  
//    double FPS = video.get(CV_CAP_PROP_FPS);//获取FPS  
//    Mat frame;//存储帧  
//    Mat temp;//存储前一帧图像  
//    Mat result;//存储结果图像  
//    for (int i = 0; i < frameCount; i++)  
//    {  
//
//        video >> frame;//读帧进frame  
//        imshow("frame", frame);  
//        if (frame.empty())//对帧进行异常检测  
//        {  
//            cout << "frame is empty!" << endl;  
//            break;  
//        }  
//        if (i == 0)//如果为第一帧（temp还为空）  
//        {  
//            result = MoveDetect(frame, frame);//调用MoveDetect()进行运动物体检测，返回值存入result  
//        }  
//        else//若不是第一帧（temp有值了）  
//        {  
//            result = MoveDetect(temp, frame);//调用MoveDetect()进行运动物体检测，返回值存入result  
//
//        }  
//        imshow("result", result);  
//        if (waitKey(1000.0 / FPS) == 27)//按原FPS显示  
//        {  
//            cout << "ESC退出!" << endl;  
//            break;  
//        }  
//        temp = frame.clone();  
//    } 
//} 
//	video.release();
	saveImage("3.avi");
	
	VideoBackground *video = new VideoBackground("image0.jpg");
	//img
	/*for (int i = 0;i < 23;i++)
	{
		char img_name[13];
		sprintf(img_name,"%s%d%s","image",i,".jpg");
		Mat img = imread(img_name);
		video1->gaussBackground(img);
	}*/
	video ->initialize();
	video ->gaussBackground("3.avi");

    return 0;  

}  
Mat MoveDetect(Mat temp, Mat frame)  
{  
    Mat result = frame.clone();  
    //1.将background和frame转为灰度图  
    Mat gray1, gray2;  
    cvtColor(temp, gray1, CV_BGR2GRAY);  
    cvtColor(frame, gray2, CV_BGR2GRAY);  
    //2.将background和frame做差  
    Mat diff;  
    absdiff(gray1, gray2, diff);  
    imshow("diff", diff);  
    //3.对差值图diff_thresh进行阈值化处理  
    Mat diff_thresh;  
    threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);  
    imshow("diff_thresh", diff_thresh);  
    //4.腐蚀  
    Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));  
    Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(18, 18));  
    erode(diff_thresh, diff_thresh, kernel_erode);  
    imshow("erode", diff_thresh);  
    //5.膨胀  
    dilate(diff_thresh, diff_thresh, kernel_dilate);  
    imshow("dilate", diff_thresh);  
    //6.查找轮廓并绘制轮廓  
    vector<vector<Point> > contours;  
    findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);  
    drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓  
    //7.查找正外接矩形  
    vector<Rect> boundRect(contours.size());  
    for (int i = 0; i < contours.size(); i++)  
    {  
        boundRect[i] = boundingRect(contours[i]);  
        rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形  
    }  
    return result;//返回result  
} 
//Mat VideoBackground()
//{

	

//}
//class VideoBackground
//{
//private:
//	string filename;
//	int k,D;           //单高斯模型的个数k，偏差阈值D
//	int width,height;
//	double alpha;       //学习速率
//	double wightThresh ;
//	double inSigma;
//	double bgTol;    
//	double ***muBlue;   //三维数组指针 蓝色通道的均值
//	double ***muGreen;
//	double ***muRed;
//	double ***w;        //初始化权重矩阵
//	double ***sigma;    //标准差矩阵
//
//public:
//	//VideoBackground();
//	VideoBackground(string file);    //构造函数初始化用于聚类的
//
//	//void saveImage() ;             //保存随机一部分帧 用于背景计算
//	Mat gaussBackground();           //计算背景  混合高斯模型
//	Mat updateBackground();          //跟新背景
//	bool absoluteDistance(int xVal, double mu, double sigma);//计算绝对距离
//	void updateWeights(int i ,int j,int l);                //跟新权重
//	void updateMeans(int i ,int j,int l,int blue,int green,int red,double rho);  //更新均值
//	void updateSigma(int i,int j,int l,int blue,int green,int red,double rho);   //更新标准差
//	void sort(int i,int j);
//};
VideoBackground::VideoBackground(string file)
{
	Mat img1;
	img1 = imread(file);
	width = img1.size().width;
	height = img1.size().height;
	//img1 = imread("image0.jpg");
	back = img1.clone();  //初始化  back
	//cvtColor(img, back, CV_BGR2GRAY);
	cvtColor(img1, fore, CV_BGR2GRAY);//初始化fore
	//video2.release();
}
//保存帧用于 训练背景
void saveImage(string file)
{
	string filename = file;
	int cntFrame = 23;
	Mat frame;
	VideoCapture video2(filename);
	if(!video2.isOpened())
	{
		printf("打开视频失败！");
	}
	else
	{
		for(int i=0;i < cntFrame;i++)
		{
			char img_name[13];
			sprintf(img_name,"%s%d%s","image",i,".jpg");//
			//string Img_name =""+to_string(i)+".bmp";
			video2 >> frame;
			imwrite(img_name,frame);
			//waitKey(10);
		}
	}
	video2.release();
}
void VideoBackground::initialize()
{
	k = 3;
	D = 2.5;
	alpha = 0.04;
	wightThresh = 0.5;
	inSigma = 11.0;
	bgTol = 30;
	//Mat back;
	//Mat fore;
	//初始化
	Mat img;
	img = imread("image0.jpg");
	muBlue = new double **[int(height)];
	muGreen = new double **[int(height)];
	muRed = new double **[int(height)];
	w = new double **[int(height)];
	sigma = new double **[int(height)];
	srand((unsigned)time(NULL));  
	for (int i = 0;i < height;i++)
	{
		muBlue[i] = new double *[width];
		muGreen[i] = new double *[width];
		muRed[i] = new double *[width];
		w[i] = new double *[width];
		sigma[i] = new double *[width];

		for (int j = 0;j < width; j++)
		{
			muBlue[i][j] = new double [k];
			muGreen[i][j] = new double [k];
			muRed[i][j] = new double [k];
			w[i][j] = new double [k];
			sigma[i][j] = new double [k];
			for (int l = 0; l < k ; l++)
			{
				Vec3b pix = img.at<Vec3b>(i,j);
				muBlue[i][j][l] = double(pix.val[0]);
				muGreen[i][j][l] = double(pix.val[1]);
				muRed[i][j][l] = double(pix.val[2]);
				//rand()%(255)
				w[i][j][l] = (double)(1.0/double(k));     //初始化第k个高斯分布的权重
				//printf("%lf\n%lf\n",(double)(1.0/double(k)),w[i][j][k]);
				sigma[i][j][l] = inSigma;               //初始化标准差
			}

		}

	}
}
//void VideoBackground::createAlphaMat(Mat &mat)
//{
//	for (int i = 0; i < mat.rows; i++)
//    {
//        for (int j = 0; j < mat.cols; j++)
//        {
//            Vec1b& rgba = mat.at<Vec1b>(i, j);
//            rgba[0] = 0;
//        }
//    }
//}
//void VideoBackground::saveImage()
//{
//	//filename
//	int cntFrame = 23;
//	Mat frame;
//	VideoCapture video("filename");
//	if(!video.isOpened())
//	{
//		printf("打开视频失败！");
//	}
//	else
//	{
//	for(int i=0;i<cntFrame;i++)
//	{
//		char img_name[13];
//		sprintf(img_name,"%s%d%s","image",i,".bmp");//
//		//string Img_name =""+to_string(i)+".bmp";
//		video >> frame;
//		imwrite(img_name,frame);
//	}
//	//return 
//	}
//}
Mat VideoBackground::gaussBackground(Mat img)
{

	//Mat img1 = imread("image1.bmp");
	//imshow("2",img);
	Mat img_back;
	/*for (int i = 0;i < 23;i++)
	{
		char img_name[13];
		sprintf(img_name,"%s%d%s","image",1,".jpg");
		Mat img = imread(img_name);*/
	int match = 0;
	//更新高斯模型的参数 
	//Mat gray;
	//cvtColor(img, gray, CV_BGR2GRAY);
	for(int i = 0;i < height;i++)
	{
		for(int j = 0;j < width;j++)
		{
			Vec3b pix = img.at<Vec3b>(i,j);
			uchar blue = pix.val[0];
			uchar green = pix.val[1];
			uchar red = pix.val[2];
			for(int l = 0;l < k;l++)
			{

				//像素与第k个高斯模型匹配
				if (absoluteDistance(blue,muBlue[i][j][l],sigma[i][j][l])||
					absoluteDistance(green,muGreen[i][j][l],sigma[i][j][l])||
					absoluteDistance(red,muRed[i][j][l],sigma[i][j][l])
					)
				{
					double p;
					match = 1;
					//更新权重
//					extern double pow(double x,double y);
					//w[i][j][l] = (1.0-alpha)*w[i][j][l] + alpha;
					updateWeights(i,j,l);
					/*p = alpha*(1.0/(pow (2.0*M_PI*sigma[i][j][l]*sigma[i][j][l], 1.5)))*
						exp(-0.5*(pow(((double)blue - muBlue[i][j][l]), double(2.0)) + 
						pow(((double)green - muGreen[i][j][l]), double(2.0)) + 
						pow(((double)red - muRed[i][j][l]), double(2.0)))/pow(sigma[i][j][l], double(2.0)));*/
					//更新均值
					p = alpha/w[i][j][l];
					//cout<<p<<endl;
					updateMeans(i,j,l,int(blue),int(green),int(green),p);
					//更新标准差
					updateSigma(i,j,l,int(blue),int(green),int(green),p);
					//对各权重 均值 标准差进行排序
					//sort(i,j);
					//计算背景和前景


				}
				else
				{
					//w[i][j][l] = (1.0-alpha)*w[i][j][l];
					updateWeights(i,j,k-1);
				}
			}
			sort(i,j);
			//计算背景
			/*for(int x = 0;x < k;x++)
			{
				back
			}*/
			if (match = 0)
			{
				//w[i][j][k-1] = 0.33/(double(k));
				muBlue[i][j][k-1] = double(blue);
				muGreen[i][j][k-1] = double(green);
				muRed[i][j][k-1] = double(red);
				sigma[i][j][k-1] = inSigma;
				//sort(i,j);
				//updateWeights(i,j,k-1);
			}
			//sort(i,j);
			//计算背景和前景
			//ForeBackground img1;
			updateBackground(i,j,img,match);
		}
	}
	imshow("background",back);
	waitKey(10);
	//medianBlur(fore, fore, 5);
	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(2, 2));  
    Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(18, 18));  
    erode(fore, fore, kernel_erode);  
	//medianBlur(fore, fore, 5);
    //imshow("erode", diff_thresh);  
    //5.膨胀  
    dilate(fore, fore, kernel_dilate); 
	imshow("foreground",fore);
	waitKey(10);
	//}
	return img_back;
}
Mat VideoBackground::gaussBackground(string videoname)
{
	while(1)
	{
		VideoCapture video1(videoname);
		if (!video1.isOpened())  //对video进行异常检测 
		{
			cout<<"打开视频失败！"<<endl;
		}
		Mat img;
		int frameCount = video1.get(CV_CAP_PROP_FRAME_COUNT);//获取帧数  
		for (int i = 0; i < frameCount; i++) 
		{
			  video1 >> img;//读帧进img 
			  imshow("frame", img);
			  //waitKey();
			  //video1 ->gaussBackground(img);
			  int match = 0;
			 for(int i = 0;i < height;i++)
			 {
				 for(int j = 0;j < width;j++)
				 {
					Vec3b pix = img.at<Vec3b>(i,j);
					uchar blue = pix.val[0];
					uchar green = pix.val[1];
					uchar red = pix.val[2];
					for(int l = 0;l < k;l++)
					{
						if (absoluteDistance(blue,muBlue[i][j][l],sigma[i][j][l])&&
							absoluteDistance(green,muGreen[i][j][l],sigma[i][j][l])&&
							absoluteDistance(red,muRed[i][j][l],sigma[i][j][l])
							)
						{
							double p;
							match = 1;
							updateWeights(i,j,l);
							p = alpha/w[i][j][l];
							updateMeans(i,j,l,int(blue),int(green),int(green),p);
							updateSigma(i,j,l,int(blue),int(green),int(green),p);
						}
						else
						{
							updateWeights(i,j,k-1);
						}
					}
					sort(i,j);
					if (match = 0)
					{
						w[i][j][k-1] = 0.33/(double(k));
						muBlue[i][j][k-1] = double(blue);
						muGreen[i][j][k-1] = double(green);
						muRed[i][j][k-1] = double(red);
						sigma[i][j][k-1] = inSigma;
						updateWeights(i,j,k-1);
					}
					updateBackground(i,j,img,match);
				}
			}
			imshow("background",back);
			//waitKey(10);
			Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(2, 2));  
			Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(3, 3)); 
			imshow("fore1",fore);
			dilate(fore, fore, kernel_dilate);
			imshow("dilate",fore);
			erode(fore, fore, kernel_erode);  
			imshow("erode",fore);
			//dilate(fore, fore, kernel_dilate);
			//imshow("dilate",fore);
			//imshow("foreground",fore);
			//waitKey(10);
			if (waitKey(1.0) == 27)//按原FPS显示  
			{  
				 cout << "ESC退出!" << endl;  
				 break;  
			}
		}
	}
}
void VideoBackground::updateBackground(int i,int j,Mat frame,int match)
{
	//ForeBackground img;
	double sum = 0;
	int x = 0;
	double bVal = 0;
	double gVal = 0;
	double rVal = 0;

	do
	{
	//for (int x = 0;x < k;x++)
	//{
		bVal += w[i][j][x]*muBlue[i][j][x];
		gVal += w[i][j][x]*muGreen[i][j][x];
		rVal += w[i][j][x]*muRed[i][j][x];
	//}
		sum += w[i][j][x];
		x++;
	}while(sum < wightThresh);
	bVal /= sum;
	gVal /= sum;
	rVal /= sum;
	back.at<Vec3b>(i,j)[0] = (bVal);
	back.at<Vec3b>(i,j)[1] = (gVal);
	back.at<Vec3b>(i,j)[2] = (rVal);
	if (match == 0){
		if (matePixel(i,j,0,bVal,frame)||
			matePixel(i,j,1,gVal,frame)||
			matePixel(i,j,2,rVal,frame)
				)
		{
			fore.at<uchar>(i,j) = 0;
		}
		else
		{
			fore.at<uchar>(i,j) = 255;
		}
	}
	//img.fore = fore.clone();
	//return img;
}
void VideoBackground::updateForeground(int i,int j)
{

}
//判断像素与第K个高斯模型匹配
bool VideoBackground::matePixel(int i,int j,int m,double Val,Mat frame)
{
	
		if(fabs(frame.at<Vec3b>(i,j)[m]-Val) <= D*sigma[i][j][0]||
			fabs(frame.at<Vec3b>(i,j)[m]-Val) <= D*sigma[i][j][1]||
		    fabs(frame.at<Vec3b>(i,j)[m]-Val) <= D*sigma[i][j][2])
		{
			return true;
		}
		else
		{
			return false;
		}
	
}
//判断像素与第m个高斯模型均值的绝对距离是否小于2.5倍的sigma
bool VideoBackground::absoluteDistance(int xVal, double mu, double sigma)
{
	//printf("%lf\n%lf\n",mu,sigma);
	if(fabs(xVal-mu)<=D*sigma)
	{
		//printf("%lf\n",fabs(xVal-mu));
		return true;
	}
	else
	{
		return false;
	}
}
//更新权重
void VideoBackground::updateWeights(int i ,int j,int l)
{
	double sum1 = 0;
	//sum1 = 0;
	for (int x = 0;x < k;x++)
	{
		if(x == l)
		{
			w[i][j][x] = (1.0-alpha)*w[i][j][x] + alpha;

		}
		else
		{
			w[i][j][x] = (1.0-alpha)*w[i][j][x];
		}
		sum1 += w[i][j][x];
	}
	//w[i][j][l] = (1.0-alpha)*w[i][j][l] + alpha;
	for(int x = 0;x < k;x++)
	{
		w[i][j][x] /= sum1;   //对每个权重进行归一化
	}

}
//更新均值
void VideoBackground::updateMeans(int i ,int j,int l,int blue,int green,int red,double rho)
{
	muBlue[i][j][l] = (1.0-rho)*muBlue[i][j][l] + rho*blue;
	muGreen[i][j][l] = (1.0-rho)*muGreen[i][j][l] + rho*green;
	muRed[i][j][l] = (1.0-rho)*muRed[i][j][l] + rho*red;
}
//更新sigma
void VideoBackground::updateSigma(int i ,int j,int l,int blue,int green,int red,double rho)
{
	sigma[i][j][l] = sqrt((1.0-rho)*pow(sigma[i][j][l], double(2.0)) + 
		(rho)*(pow(((double)blue - muBlue[i][j][l]),double(2.0))+
		pow(((double)green - muGreen[i][j][l]),double(2.0)) +
		pow(((double)red - muRed[i][j][l]),double(2.0))));
}
//对均值，sigma，权重按照从大到小进行排序
void VideoBackground::sort(int i,int j)
{
	for (int n = 1;n < k;n++)
	{
		for (int m = 0;m < k-n;m++)
		{
			if (w[i][j][m] < w[i][j][m+1])
			{
				double temp = w[i][j][m];
				w[i][j][m] = w[i][j][m+1];
				w[i][j][m+1]= temp;

				temp = muBlue[i][j][m];
				muBlue[i][j][m] = muBlue[i][j][m+1];
				muBlue[i][j][m+1]= temp;

				temp = muGreen[i][j][m];
				muGreen[i][j][m] = muGreen[i][j][m+1];
				muGreen[i][j][m+1]= temp;

				temp = muRed[i][j][m];
				muRed[i][j][m] = muRed[i][j][m+1];
				muRed[i][j][m+1]= temp;

				temp = sigma[i][j][m];
				sigma[i][j][m] = sigma[i][j][m+1];
				sigma[i][j][m+1]= temp;
			}
		}
	}
}
