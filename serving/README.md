# serving
## 安装
### docker [hub](https://hub.docker.com/r/paddlepaddle/serving/tags)

拉取镜像

`nvidia-docker pull hub.baidubce.com/paddlepaddle/serving:latest-cuda10.0-cudnn7-devel`

创建镜像

`nvidia-docker build -t hub.baidubce.com/paddlepaddle/serving:latest-cuda10.0-cudnn7-devel`

启动

`
nvidia-docker run -p 9292:9292 --name serv -dit hub.baidubce.com/paddlepaddle/serving:latest-cuda10.0-cudnn7-devel
nvidia-docker exec -it serv bash
`

