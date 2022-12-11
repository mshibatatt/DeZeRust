FROM ubuntu:latest

ENV SHELL /bin/bash
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    cmake git wget curl graphviz && \
    # install Rust
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    chmod -R +rx /root && \ 
    # clean-up
    apt-get autoremove -y && \
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

# setup Rust
# need "rustup update stable" in container
ENV PATH="/root/.cargo/bin:$PATH"

CMD ["/bin/bash"]