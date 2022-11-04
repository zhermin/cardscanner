all:
	g++ main.cpp houghlines/houghlines.cpp -o main.o -std=c++14

opencv:
	g++ main-cv.cpp \
		houghlines/houghlines.cpp \
		card_corner_detector.cpp \
		utils.cpp \
		-o main-cv.o \
		-std=c++14 `pkg-config --cflags --libs opencv4`