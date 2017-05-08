#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"

enum ANALYSIS {MOG = 1, MOG2 = 2, HAAR = 3};
enum VIDEO {NATURAL = 1, NATURAL_ARTIFICIAL = 2, ARTIFICAL = 3, DARK = 4, OUTSIDE = 5, CUSTOM = 6};

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

void do_MOG2_detection(bool* initialized, cv::Mat* frame, cv::Mat* fg_mask, cv::Mat* dup_fg_mask,  cv::Ptr<cv::BackgroundSubtractor>* bg_sub, bool reduce_noise){
    if (!*initialized) {
        std::cout << "Initializing MO2G detection" << std::endl;
        *bg_sub = cv::createBackgroundSubtractorMOG2(500,16,true);
        *initialized = true;
    }
    cv::Ptr<cv::BackgroundSubtractor> subtractor = *bg_sub;
    subtractor->apply(*frame, *fg_mask); // update background model

    if (reduce_noise) {
        // Reduce noise with morphological transformation
        cv::Mat kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(6,6));
        cv::morphologyEx(*fg_mask,*dup_fg_mask,CV_MOP_OPEN,kernel);
    } else {
        *dup_fg_mask = *fg_mask;
    }

    std::vector<std::vector<cv::Point>> contours = find_contours(dup_fg_mask);
    std::vector<cv::Rect> rects = calc_bounding_rects(&contours);
    cv::drawContours(*frame, contours, -1, cv::Scalar(0,0,255));
    draw_rects(rects, frame);
    std::cout << "here" << std::endl;
}

void do_HAAR_detection(bool* initialized, cv::Mat* frame, cv::Mat* gray_frame, cv::CascadeClassifier* cascade) {

    std::vector<cv::Rect> faces;
    cv::cvtColor(*frame, *gray_frame, CV_BGR2GRAY);
    cv::equalizeHist(*gray_frame, *gray_frame);

    cascade->detectMultiScale(*gray_frame, faces, 1.1, 3, 0, cv::Size(1,1), cv::Size(50,50));

    for (size_t i = 0; i < faces.size(); i++) {
        cv::Point upperLeft(faces[i].x, faces[i].y);
        cv::Point lowerRight(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        cv::Rect r = cv::Rect(upperLeft, lowerRight);
    }
    draw_rects(faces, frame, 1);
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

void analysis_loop(cv::VideoCapture* capture, unsigned int analysis_type, bool show_playback, bool noise_reduction, std::string video_path, bool save_frames=false) {
    cv::Mat current_frame, fg_mask, dup_fg_mask;
    cv::Ptr<cv::BackgroundSubtractor> bg_sub;
    cv::CascadeClassifier cascade;
    cascade.load("haarcascade_frontalface_default.xml");
    bool abort = false;
    bool initialized = false;
    bool reduce_noice = false;
    std::string folder = video_path.substr(0, video_path.length() - 4);
    unsigned int frame_index = 0;
    unsigned int total_num_frames = capture->get(CV_CAP_PROP_FRAME_COUNT) - 1;
    std::map<std::string, cv::Mat> frames;
    int key_code;
    int correct = 0;
    int wrong = 0;
    int both = 0;

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
                do_MOG2_detection(&initialized, &current_frame, &fg_mask, &dup_fg_mask, &bg_sub, noise_reduction);
                break;
            case HAAR:
                do_HAAR_detection(&initialized, &current_frame, &fg_mask, &cascade);
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
            if (noise_reduction) {
                cv::imshow("Modified foreground mask", dup_fg_mask);
            }
            if (frame_index % 10 == 0) {
            key_code = cv::waitKey(0);
            switch (key_code) {
                case 32: // save frames
                    frames[folder + "ori_" + std::to_string(frame_index) + ".png"] = current_frame;
                    frames[folder + "fg_" + std::to_string(frame_index) + ".png"] = fg_mask;
                    if (noise_reduction) {
                        frames[folder + "fg_dup_" + std::to_string(frame_index) + ".png"] = dup_fg_mask;
                    }
                    for (auto &ent : frames) {
                        cv::imwrite(ent.first, ent.second, {CV_IMWRITE_PNG_COMPRESSION, 9});
                    }
                    break;
                case 49: //correct
                    ++correct;
                    break;
                case 50: // wrong
                    ++wrong;
                    break;
                case 51: // both
                    ++both;
                    break;
                default:
                    break;
            }
            }

            std::cout << key_code << std::endl;
        }

        if (save_frames && frame_index == 222) {
            frames["ori_" + std::to_string(frame_index) + ".png"] = current_frame;
            frames["fg_" + std::to_string(frame_index) + ".png"] = fg_mask;
            if (noise_reduction) {
                frames["fg_dup_" + std::to_string(frame_index) + ".png"] = dup_fg_mask;
            }
            for (auto &ent : frames) {
                cv::imwrite(ent.first, ent.second, {CV_IMWRITE_PNG_COMPRESSION, 9});
            }


        }
        std::cout << frame_index << std::endl;
        ++frame_index;
    }
    std::cout << "Correct: " << correct << std::endl;
    std::cout << "Wrong: " << wrong <<std::endl;
    std::cout << "Both: " << both << std::endl;
    cv::destroyAllWindows();
}

void start_analysis(std::string file_path, unsigned int analysis_type, bool show_playback, bool noise_reduction, bool save_frames) {
    cv::VideoCapture *capture = new cv::VideoCapture(file_path);
    if (capture->isOpened()) {
        std::cout << "Found video. Starting analysis." << std::endl;
        analysis_loop(capture, analysis_type, show_playback, noise_reduction, file_path, save_frames);
        capture->release();
    } else {
        std::cout << "Could not open video. Exiting." << std::endl;
    }

}

int main(int argc, char *argv[]) {
    unsigned int video_choice = 0;
    unsigned int analysis_choice = 0;
    bool show_playback = true;
    bool noise_reduction = false;
    bool save_frames = false;
    std::string video_path;

    std::cout << "(1) Natural light" << std::endl << "(2) Natural and artifical light" << std::endl << "(3) Artifial light" << std::endl << "(4) Dark" << std::endl << "(5) Outside" << std::endl << "(6) Custom" << std::endl << "Select video: ";
    std::cin >> video_choice;

    switch (video_choice) {
        case NATURAL:
            video_path = "natural_light.mp4";
            break;
        case NATURAL_ARTIFICIAL:
            video_path = "natural_and_artifical_light.mp4";
            break;
        case ARTIFICAL:
            video_path = "artifical_light.mp4";
            break;
        case DARK:
            video_path = "dark.mp4";
            break;
        case OUTSIDE:
            video_path = "outside.mp4";
            break;
        case CUSTOM:
            std::cout << "Enter the file path: ";
            std::cin >> video_path;
            break;
        default:
            std::cout << "Not a valid choice. Exiting." << std::endl;
            return 0;
    }

    std::cout << "(1) MOG" << std::endl << "(2) MOG2" << std::endl << "(3) HAAR" <<std::endl <<"Select analysis type: ";
    std::cin >> analysis_choice;
    std::cout << std::endl;
    std::cout << "(0) No" << std::endl << "(1) Yes" << std::endl <<"Use noise reduction: ";
    std::cin >> noise_reduction;
    std::cout << std::endl;
    std::cout << "(0) No" << std::endl << "(1) Yes" << std::endl <<"Save frames: ";
    std::cin >> save_frames;
    std::cout << std::endl;
    start_analysis(video_path, analysis_choice, show_playback, noise_reduction, save_frames);
    return 0;
}


