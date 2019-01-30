#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace ml;
using namespace std;

float round_float(const float input)
{
    return floor(input + 0.5f);
}

void add_noise(Mat &mat, float scale)
{
    for (int j = 0; j < mat.rows; j++)
    {
        for (int i = 0; i < mat.cols; i++)
        {
            float noise = static_cast<float>(rand() % 256);
            noise /= 255.0f;

            mat.at<float>(j, i) = (mat.at<float>(j, i) + noise*scale) / (1.0f + scale);

            if (mat.at<float>(j, i) < 0)
                mat.at<float>(j, i) = 0;
            else if (mat.at<float>(j, i) > 1)
                mat.at<float>(j, i) = 1;
        }
    }
}


int main(int argc, char *argv[])
{
        const int image_width = 64;
        const int image_height = 64;

        // Read in 64 row x 64 column images
        Mat dove = imread("dove.png", IMREAD_GRAYSCALE);
        Mat flowers = imread("flowers.png", IMREAD_GRAYSCALE);
        Mat peacock = imread("peacock.png", IMREAD_GRAYSCALE);
        Mat statue = imread("statue.png", IMREAD_GRAYSCALE);

        // Reshape from 64 rows x 64 columns image to 1 row x (64*64) columns
        dove = dove.reshape(0, 1);
        flowers = flowers.reshape(0, 1);
        peacock = peacock.reshape(0, 1);
        statue = statue.reshape(0, 1);

        // Convert CV_8UC1 to CV_32FC1
        Mat flt_dove(dove.rows, dove.cols, CV_32FC1);

        for (int j = 0; j < dove.rows; j++)
            for (int i = 0; i < dove.cols; i++)
                flt_dove.at<float>(j, i) = dove.at<unsigned char>(j, i) / 255.0f;

        Mat flt_flowers(flowers.rows, flowers.cols, CV_32FC1);

        for (int j = 0; j < flowers.rows; j++)
            for (int i = 0; i < flowers.cols; i++)
                flt_flowers.at<float>(j, i) = flowers.at<unsigned char>(j, i) / 255.0f;

        Mat flt_peacock(peacock.rows, peacock.cols, CV_32FC1);

        for (int j = 0; j < peacock.rows; j++)
            for (int i = 0; i < peacock.cols; i++)
                flt_peacock.at<float>(j, i) = peacock.at<unsigned char>(j, i) / 255.0f;

        Mat flt_statue = Mat(statue.rows, statue.cols, CV_32FC1);

        for (int j = 0; j < statue.rows; j++)
            for (int i = 0; i < statue.cols; i++)
                flt_statue.at<float>(j, i) = statue.at<unsigned char>(j, i) / 255.0f;

        Ptr<ANN_MLP> mlp = ANN_MLP::create();

        // Neural network elements
        const int num_input_neurons = dove.cols; // One input neuron per grayscale pixel
        const int num_output_neurons = 2; // 4 images to classify, so number of bits needed is ceiling(ln(n)/ln(2))
        const int num_hidden_neurons = static_cast<int>(sqrtf(image_width*image_height*num_output_neurons));
        Mat layersSize = Mat(3, 1, CV_16UC1);
        layersSize.row(0) = Scalar(num_input_neurons);
        layersSize.row(1) = Scalar(num_hidden_neurons);
        layersSize.row(2) = Scalar(num_output_neurons);
        mlp->setLayerSizes(layersSize);

        // Set various parameters
        mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);
        TermCriteria termCrit = TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS, 1, 0.000001);
        mlp->setTermCriteria(termCrit);
        mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP, 0.0001);

        Mat output_training_data = Mat(1, num_output_neurons, CV_32FC1).clone();

        // Train the network once
        output_training_data.at<float>(0, 0) = -1;
        output_training_data.at<float>(0, 1) = -1;
        Ptr<TrainData> trainingData = TrainData::create(flt_dove, SampleTypes::ROW_SAMPLE, output_training_data);
        mlp->train(trainingData, ANN_MLP::TrainFlags::NO_INPUT_SCALE | ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);

        // Train the network again and again
        for (int i = 1; i < 1000; i++)
        {
            if (i % 100 == 0)
                cout << i << endl;

            // Make noisy version of images to be used as network input
            Mat flt_dove_noise = flt_dove.clone();
            add_noise(flt_dove_noise, 0.1f);


            /*imshow("flt_dove_noise",flt_dove_noise.reshape(1,64));
            waitKey(0);*/

            // Train for image 1
            output_training_data.at<float>(0, 0) = -1;
            output_training_data.at<float>(0, 1) = -1;
            trainingData = TrainData::create(flt_dove_noise, SampleTypes::ROW_SAMPLE, output_training_data);
            mlp->train(trainingData, ANN_MLP::TrainFlags::UPDATE_WEIGHTS | ANN_MLP::TrainFlags::NO_INPUT_SCALE | ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);
        }

        int num_tests = 100;
        int num_successes = 0;
        int num_failures = 0;

        // Test the network again and again
        for (int i = 0; i < num_tests; i++)
        {
            // Use noisy input images
            Mat flt_dove_noise = flt_dove.clone();
            add_noise(flt_dove_noise, 0.1f);

            Mat result;
            mlp->predict(flt_dove_noise, result);

            if ((result.at<float>(0, 0)) < 0 && (result.at<float>(0, 1) < 0))
            {
                num_successes++;
            }
            else
            {
                num_failures++;
            }

        }

        cout << "Success rate: " << 100.0*static_cast<double>(num_successes) / static_cast<double>(num_tests) << "%" << endl;

        return 0;
}
