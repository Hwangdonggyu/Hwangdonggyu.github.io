---
title: FreeSurfer 설치하기
date: 2024-05-08 13:00:00 +0900
categories: [AI,freesurfer]
tags: [AI,freesurfer]
---

MRI 데이터 분석에 필요한 FreeSurfer 설치에 대해 적어보려한다.

## FreeSurefer 설치
- [FreeSurfer 설치 관련 유튜브](https://www.youtube.com/watch?v=BSQUVktXTzo&list=PLIQIswOrUH6_DWy5mJlSfj6AWY0y9iUce&index=2)
- [FreeSurfer download document](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)

FreeSurfer는 Mac과 Linux OS에서만 지원을 한다. 따라서 다른 OS를 이용중이라면 가상환경을 설치하는 것을 추천한다.

Linux 사용자 기준으로 작성을 해보려한다. 만약 Mac를 이용하고 있다면 [Mac.ver](https://surfer.nmr.mgh.harvard.edu/fswiki//FS7_mac)을 참고하면 좋을 것 같다.

### freesurfer 압축 파일 다운로드

![](https://velog.velcdn.com/images/acadias12/post/4f1c8a00-fc7f-481f-aad0-fcf7de887dbe/image.png)

2024-05-08월 기준 7.4.1 release를 다운받으면 된다. 본인은 위의 사진 맨위에 압축파일을 다운받았다.


### 압축 풀기
본인이 사용하는 환경에서 압축을 풀어야한다.

```bash
$ cd $home
$ pwd
/home/tester
```

**tar.gz 압축 풀기**
```bash
$ tar -zxpf freesurfer-linux-centos7_x86_64-7.4.1.tar.gz
```

압축이 풀렸다면 freesurfer 폴더가 생겼을 것이다.

```bash
$ cd freesurfer
$ pwd
/home/tester/freesurfer
```

### FreeSurfer 실행

```bash
$ export FREESURFER_HOME=$HOME/freesurfer
$ source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
해당 코드는 FreeSurfer 공식 문서를 참고한 코드이다. 아마 SetUpFreeSurfer라 되어있지 않은 경우도 있을 것이다.

```bash
ls
```
ls 명령어를 통해 SetUpFreeSurfer.sh가 이름이 어떻게 되어있는지 확인해보고 
```bash
$ source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
SetUpFreeSurfer.sh에 해당하는 이름으로 바꿔주면 된다.

### Freesurfer 실행화면

```bash
-------- freesurfer-linux-centos7_x86_64-7.2.0-20210720-aa8f76b --------
Setting up environment for FreeSurfer/FS-FAST (and FSL)
FREESURFER_HOME   /home/tester/freesurfer
FSFAST_HOME       /home/tester/freesurfer/fsfast
FSF_OUTPUT_FORMAT nii.gz
SUBJECTS_DIR      /home/tester/freesurfer/subjects
MNI_DIR           /home/tester/freesurfer/mni
```

위의 나와 있는 경로가 설정되어있을 것이다. ex) cd $FREESURFER_HOME

### .bashrc setup (optional)
```bash
.bashrc 셋업하는법 - freesurfer 예시
1. vim .bashrc (or nano .bashrc or gedit .bashrc)
2. 아래 코드 추가 (path는 예시입니다)
alias fs='export FREESURFER_HOME=/autofs/cluster/freesurfer/centos7_x86_64/7.4.1 &&\
        export FSFAST_HOME=$FREESURFER_HOME/fsfast &&\
        export SUBJECTS_DIR=$FREESURFER_HOME/subjects &&\
        export MNI_DIR=$FREESURFER_HOME/mni &&\
        source $FREESURFER_HOME/SetUpFreeSurfer.sh' # type 'fs' when I need freesurfer env
```
밑의 코드를 통해 fs만 치면 위의 실행과정을 해주지 않아도 된다.