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

void draw_rects(std::vector<cv::Rect> rects, cv::Mat* frame, unsigned int min_area=100) {
    for (cv::Rect r : rects) {
        if (r.area() > min_area) {
            cv::rectangle(*frame, r, cv::Scalar(0,255,0));
        }
    }
}

std::vector<cv::Rect> calc_bounding_rects(std::vector<std::vector<cv::Point>>* contours) {
    std::vector<cv::Rect> bounding_rects;
    for (std::vector<cv::Point> contour : *contours) {
        bounding_rects.push_back(cv::boundingRect(contour));
    }
    return bounding_rects;
}

std::vector<std::vector<cv::Point>> find_contours(cv::Mat* search_frame) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(*search_frame, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    return contours;
}

void do_MOG_detection(bool* initialized, cv::Mat* frame, cv::Mat* fg_mask, cv::Mat* dup_fg_mask,  cv::Ptr<cv::BackgroundSubtractor> bg_sub){
    if (!*initialized) {
        std::cout << "Initializing MOG detection" << std::endl;
        //bg_sub = cv::BackgroundSubtractorMOG();
        *initialized = true;
    }
}

void do_MOG2_detection(bool* initialized, cv::Mat* frame, cv::Mat* fg_mask, cv::Mat* dup_fg_mask,  cv::Ptr<cv::BackgroundSubtractor>* bg_sub, bool shadow_detect=true){
    if (!*initialized) {
        std::cout << "Initializing MO2G detection" << std::endl;
        *bg_sub = cv::createBackgroundSubtractorMOG2(500,16,shadow_detect);
        *initialized = true;
    }
    cv::Ptr<cv::BackgroundSubtractor> subtractor = *bg_sub;
    subtractor->apply(*frame, *fg_mask); // update background model

    // Reduce noise with morphological transformation
    cv::Mat kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(10,10));
    cv::morphologyEx(*fg_mask,*dup_fg_mask,CV_MOP_OPEN,kernel);


    std::vector<std::vector<cv::Point>> contours = find_contours(dup_fg_mask);
    std::vector<cv::Rect> rects = calc_bounding_rects(&contours);
    cv::drawContours(*frame, contours, -1, cv::Scalar(0,0,255));
    draw_rects(rects, frame);
}

void add_frame_index_overlay(cv::Mat* frame, unsigned int current_index, unsigned int last_index) {
    std::string c_index = "Current: " + std::to_string(current_index);
    std::string l_index = "Total: " + std::to_string(last_index);
    std::vector<std::string> overlay_strings = {"Frames: ", l_index, c_index};
    unsigned int width_indent = 15;
    unsigned int height_indent = 25;

    cv::rectangle(*frame, cv::Point(width_indent, height_indent/2-5), cv::Point(width_indent+150, height_indent*overlay_strings.size()+5), cv::Scalar(255,255,255), -1);
    for (unsigned int i = 0; i < overlay_strings.size(); ++i) {
        cv::putText(*frame, overlay_strings.at(i), cv::Point(width_indent,height_indent*(i+1)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0));
    }
}

void analysis_loop(cv::VideoCapture* capture, unsigned int analysis_type, bool show_playback, bool save_frames=false) {
    cv::Mat current_frame, fg_mask, dup_fg_mask;
    cv::Ptr<cv::BackgroundSubtractor> bg_sub;
    bool abort = false;
    bool initialized = false;
    unsigned int frame_index = 0;
    unsigned int total_num_frames = capture->get(CV_CAP_PROP_FRAME_COUNT) - 1;

    if (show_playback) {
        cv::namedWindow("Video");
        cv::namedWindow("Foreground mask");
        cv::namedWindow("Modified foreground mask");
    }

    std::cout << "Starting analysis" << std::endl;
    while (!abort && capture->read(current_frame)) {
        switch (analysis_type) {
            case MOG:
                do_MOG_detection(&initialized, &current_frame, &fg_mask, &dup_fg_mask, bg_sub);
                break;
            case MOG2:
                do_MOG2_detection(&initialized, &current_frame, &fg_mask, &dup_fg_mask, &bg_sub);
                break;
            default:
                std::cout << "Non-valid analysis type, aborting." << std::endl;
                abort = true;
                break;
        }
        if (show_playback) {
            add_frame_index_overlay(&current_frame, frame_index, total_num_frames - 1);
            cv::imshow("Video", current_frame);
            cv::imshow("Foreground mask", fg_mask);
            cv::imshow("Modified foreground mask", dup_fg_mask);
            cv::waitKey(50);
        }

        if (save_frames && frame_index % 20 == 0) {
            std::map<std::string, cv::Mat> frames;
            frames["ori_" + std::to_string(frame_index) + ".png"] = current_frame;
            frames["fg_" + std::to_string(frame_index) + ".png"] = fg_mask;
            frames["fg_dup_" + std::to_string(frame_index) + ".png"] = dup_fg_mask;
            for (auto &ent : frames) {
                cv::imwrite(ent.first, ent.second, {CV_IMWRITE_PNG_COMPRESSION, 9});
            }


        }
        ++frame_index;
    }
    cv::destroyAllWindows();
}

void start_analysis(std::string file_path, unsigned int analysis_type, bool show_playback) {
    cv::VideoCapture *capture = new cv::VideoCapture(file_path);
    if (capture->isOpened()) {
        std::cout << "Found video. Starting analysis." << std::endl;
        analysis_loop(capture, analysis_type, show_playback, true);
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
            video_path = "vid_01.mp4";
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


