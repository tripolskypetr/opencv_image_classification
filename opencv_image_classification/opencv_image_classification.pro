QT += core

CONFIG += console

TARGET = opencv_image_classification
TEMPLATE = app

DEFINES += QT_DEPRECATED_WARNINGS

CONFIG += c++11

SOURCES += \
        main.cpp

HEADERS +=


win32 {
    INCLUDEPATH += "C:\\opencv\\build\\include" \

    CONFIG(debug) {
        LIBS += -L"C:\\opencv\\build\\x64\\vc15\\lib" \
            -lopencv_world401d
    }

    CONFIG(release) {
        LIBS += -L"C:\\opencv\\build\\x64\\vc15\\lib" \
            -lopencv_world401
    }
}
