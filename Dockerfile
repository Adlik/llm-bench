# If you need to test other models or use a different runtime environment
# please modify the base image in the Dockerfile and install the corresponding dependencies accordingly.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    python3 \
    python3-distutils \
    python3-pip \
    unzip \
    zip \
    openssh-server \
    htop \
    git \
    curl \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip accelerate
RUN python3 -m pip install argparse
RUN python3 -m pip install requests
RUN python3 -m pip install pandas
RUN python3 -m pip install google
RUN python3 -m pip install numpy
RUN python3 -m pip install transformers
RUN python3 -m pip install sentencepiece


RUN apt-get update -y
RUN apt-get install -y build-essential
RUN useradd -m -u 1000 babelcode

# Installing Julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz \
    && tar zxvf julia-1.7.3-linux-x86_64.tar.gz

ENV PATH="/julia-1.7.3/bin:$PATH"

# Installing Java
RUN apt-get install -y default-jre=2:1.11*

# Installing Go
RUN wget https://go.dev/dl/go1.19.linux-amd64.tar.gz
RUN tar -C /usr/local -xvf go1.19.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin

# Installing Node and NPM. Use the NVM package manager to ensure specific node
# versions.
ENV NVM_DIR=/nvm
RUN mkdir -p "${NVM_DIR}"
ENV NODE_VERSION=16.13.0
RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
RUN . "${NVM_DIR}/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "${NVM_DIR}/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "${NVM_DIR}/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="${NVM_DIR}/versions/node/v${NODE_VERSION}/bin/:${PATH}"

# # Installing Lua
RUN apt install -y lua5.3

# # Installing Kotlin
RUN wget https://github.com/JetBrains/kotlin/releases/download/v1.7.10/kotlin-compiler-1.7.10.zip
RUN unzip kotlin-compiler-1.7.10.zip
ENV PATH="/kotlinc/bin:$PATH"
RUN mkdir /kotlin_dir
ENV KONAN_DATA_DIR="/kotlin_dir"
RUN chown -R 1000:root "$KONAN_DATA_DIR" && chmod -R 775 "$KONAN_DATA_DIR"
# Need to do this so it downloads necessary dependencies
RUN echo 'fun main() {println("Hello Kotlin/Native!")}' > code.kt 
RUN kotlinc code.kt -include-runtime -d test.jar
RUN rm code.kt

# Installing Rust
RUN wget https://static.rust-lang.org/dist/rust-1.64.0-x86_64-unknown-linux-gnu.tar.gz
RUN tar -xvf rust-1.64.0-x86_64-unknown-linux-gnu.tar.gz
RUN bash rust-1.64.0-x86_64-unknown-linux-gnu/install.sh

# Installing Haskell
RUN apt-get  -y install haskell-platform=2014.*

# Installing C#
RUN wget https://dot.net/v1/dotnet-install.sh
RUN chmod +x ./dotnet-install.sh
RUN ./dotnet-install.sh -c 6.0 --runtime aspnetcore
RUN apt-get install -y mono-complete --fix-missing || true

# Install PHP
RUN apt-get install -y php-cli

# Install Scala
RUN wget https://github.com/coursier/coursier/releases/download/v2.1.0-RC5/cs-x86_64-pc-linux-static.gz 
RUN gzip -d cs-x86_64-pc-linux-static && mv cs-x86_64-pc-linux-static cs && chmod +x cs
RUN ./cs setup -y || true
RUN mv ~/.local/share/coursier /coursier || true
RUN chown -R 1000:root "/coursier" && chmod -R 775 "/coursier" || true
ENV PATH="$PATH:/coursier/bin"

# Install R
RUN apt-get install r-base -y --fix-missing || true

# Install dart
RUN wget https://storage.googleapis.com/dart-archive/channels/stable/release/2.18.5/linux_packages/dart_2.18.5-1_amd64.deb
RUN dpkg -i dart_2.18.5-1_amd64.deb
RUN apt-get install -f
RUN mv "/usr/lib/dart" "/dart"
RUN chown -R 1000:root "/dart" && chmod -R 775 "/dart"
ENV PATH="$PATH:/dart/bin"

RUN python3 -m pip install pytest==7.1.2
RUN python3 -m pip install jinja2==3.1.2
RUN python3 -m pip install pyyaml==6.0
RUN python3 -m pip install gin-config==0.5.0
RUN python3 -m pip install coverage==6.4.4
RUN python3 -m pip install psutil==5.9.2
RUN python3 -m pip install absl-py==1.2.0
RUN python3 -m pip install tensorflow==2.10.0
RUN python3 -m pip install astor

WORKDIR /
RUN python3 -m pip install human-eval

RUN git clone https://github.com/google-research/babelcode.git
COPY ./fixs/human_eval_cn.jsonl /babelcode/data/dataset_fixes
COPY ./fixs/human_eval_en.jsonl /babelcode/data/dataset_fixes
COPY ./fixs/utils.py /babelcode/babelcode/dataset_conversion/utils.py
COPY ./fixs/__init__.py /babelcode/__init__.py

ENV ALLOW_EXECUTION true

COPY . /llm-bench

WORKDIR /llm-bench
