#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

enum ANALYSIS {MOG = 1, MOG2 = 2};
enum VIDEO {DEFAULT = 1, CUSTOM = 2};

// Global vars



void do_MOG_detection(bool* initialized, cv::Mat* frame, cv::Mat* fg_mask, cv::Ptr<cv::BackgroundSubtractor> bg_sub){
    if (!*initialized) {
        std::cout << "Initializing MOG detection" << std::endl;
        //bg_sub = cv::BackgroundSubtractorMOG();
        *initialized = true;
    }
}

void do_MOG2_detection(bool* initialized, cv::Mat* frame, cv::Mat* fg_mask, cv::Ptr<cv::BackgroundSubtractor> bg_sub){
    if (!*initialized) {
        std::cout << "Initializing MO2G detection" << std::endl;
        bg_sub = cv::createBackgroundSubtractorMOG2();
        *initialized = true;
    }
}

void analysis_loop(cv::VideoCapture* capture, unsigned int analysis_type, bool show_playback) {
    cv::Mat current_frame, fg_mask;
    cv::Ptr<cv::BackgroundSubtractor> bg_sub;
    bool abort = false;
    bool initialized = false;
    unsigned int frame_index = 0;

    if (show_playback) {
        cv::namedWindow("Video");
    }

    std::cout << "Starting analysis" << std::endl;
    while (!abort && capture->read(current_frame)) {
        switch (analysis_type) {
            case MOG:
                do_MOG_detection(&initialized, &current_frame, &fg_mask, bg_sub);
                break;
            case MOG2:
                do_MOG2_detection(&initialized, &current_frame, &fg_mask, bg_sub);
                break;
            default:
                std::cout << "Non-valid analysis type, aborting." << std::endl;
                abort = true;
                break;
        }
        if (show_playback) {
            cv::imshow("Video", current_frame);
            cv::waitKey(24);
        }
        ++frame_index;
    }
    cv::destroyAllWindows();
}

void start_analysis(std::string file_path, unsigned int analysis_type, bool show_playback) {
    cv::VideoCapture *capture = new cv::VideoCapture(file_path);
    if (capture->isOpened()) {
        std::cout << "Found video. Starting analysis." << std::endl;
        analysis_loop(capture, analysis_type, show_playback);
        capture->release();
    } else {
        std::cout << "Could not open video. Exiting." << std::endl;
    }

}

int main(int argc, char *argv[]) {
    unsigned int video_choice = 0;
    unsigned int analysis_choice = 0;
    bool show_playback = true;
    std::string video_path;

    std::cout << "(1) Default" << std::endl << "(2) Custom" << std::endl << "Select video: ";
    std::cin >> video_choice;

    switch (video_choice) {
        case DEFAULT:
            video_path = "seq_01.mp4";
            break;
        case CUSTOM:
            std::cout << "Enter the file path: ";
            std::cin >> video_path;
            break;
        default:
            std::cout << "Not a valid choice. Exiting." << std::endl;
            return 0;
    }

    std::cout << "(1) MOG" << std::endl << "(2) MOG2" << std::endl <<"Select analysis type: ";
    std::cin >> analysis_choice;
    std::cout << std::endl;

    start_analysis(video_path, analysis_choice, show_playback);
    return 0;
}


