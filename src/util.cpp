#include "libs.h"
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

std::string chooseImage(const std::string& directory) {
    std::vector<std::string> images;

    std::cout << "Scan dir: " << directory << std::endl;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                images.push_back(entry.path().string());
            }
        }
    }

    if (images.empty()) {
        std::cout << "No images here :(." << std::endl;
        return "";
    }

    std::cout << "Some images here:\n";
    for (size_t i = 0; i < images.size(); ++i) {
        std::cout << i + 1 << ") " << fs::path(images[i]).filename().string() << std::endl;
    }

    int choice = 0;
    while (true) {
        std::cout << "\nSelect image: ";
        std::cin >> choice;

        if (choice >= 1 && choice <= (int)images.size()) break;
        std::cout << "Incorrect. Try again!" << std::endl;
    }

    std::cout << "Selected: " << images[choice - 1] << "\n";
    return images[choice - 1];
}