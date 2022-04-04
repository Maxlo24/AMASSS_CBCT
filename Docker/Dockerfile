FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ARG RELEASE_VERSION="v1.0.0-alpha"
ARG RELEASE="1.0.0-alpha"

RUN apt-get update && \
	apt-get install -y git \
	libx11-6 \
	libgl1 \
	libopengl0 \
	libegl1 \
	wget\
	-y unzip

	

RUN mkdir /app
WORKDIR /app
RUN wget https://github.com/Maxlo24/AMASSS_CBCT/archive/refs/tags/$RELEASE_VERSION.zip
RUN unzip $RELEASE_VERSION.zip
# RUN unzip AMASSS_CBCT-$RELEASE.zip
# RUN unzip AMASSS_CBCT-1.0.0-alpha.zip
RUN mv AMASSS_CBCT-$RELEASE/MULTI_SEG MULTI_SEG
# RUN mv /app/$RELEASE_VERSION/Data /app/Data
RUN rm -rf $RELEASE_VERSION.zip \
	rm -rf $RELEASE_VERSION

RUN pip install -r /app/MULTI_SEG/requirements.txt


RUN wget https://github.com/Maxlo24/AMASSS_CBCT/releases/download/$RELEASE_VERSION/ALL_MODELS.zip
RUN unzip ALL_MODELS.zip
RUN rm -rf ALL_MODELS.zip

RUN mkdir data


