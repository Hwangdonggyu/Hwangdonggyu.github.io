---
title: Node.js에 대한 기초
date: 2023-05-02 12:35:00 +0900
categories: [Node.js]
tags: [Node.js]
---

![](https://velog.velcdn.com/images/acadias12/post/05fa6d48-dfbf-4790-817d-6ec18672af5b/image.png)

최근 백엔드 개발에 관심을 가지게 되어 Node.js에 대해 공부하기 시작하였는데, Node.js에 대한 기초 지식에 대하여 정리해보려 한다.

## Node.js의 개념
> Node.js는 Chrome V8 JavaScript 엔진으로 빌드 된 JavaScript 런타임 환경이다.

Node.js는 이해하기 쉬운 코드를 사용하여 빠르고 확장 가능한 네트워크 애플리케이션을 구축할 수 있다. Windows OS, Mac OSX, Linux, Unix 및 기타 운영 체제에서 실행된다.

## Node.js는 서버일까?
> Node.js는 서버가 아닌 JavaScript 엔진으로 빌드 된 JavaScript 런타임 환경이다.

Node.js를 서버 혹은 프레임 워크라고 오해받는 경우가 많지만, Node.js는 JavaScript의 실행 환경이다.


## Node.js의 장단점에 대해 알아보자
- 장점

    1. 빠른 실행 속도: Node.js는 Google V8 엔진을 사용하므로, JavaScript 코드의 실행 속도가 매우 빠르다.

    2. 비동기식 I/O 처리: Node.js는 비동기식 I/O 처리를 지원하므로, 클라이언트 요청 및 데이터베이스 처리와 같은 I/O 작업을 효율적으로 처리할 수 있다.

    3. 다양한 모듈 시스템: Node.js는 NPM(Node Package Manager)을 기반으로 다양한 모듈 시스템을 사용하여 라이브러리 및 모듈을 쉽게 추가하고 관리할 수 있다.

    4. 높은 확장성: Node.js는 클러스터링 및 로드 밸런싱과 같은 기능을 지원하여, 서버의 확장성을 높일 수 있다.

- 단점

    1. 단일 스레드 모델: Node.js는 단일 스레드 모델을 사용하기 때문에, CPU 집약적인 작업에 대해서는 성능이 저하될 수 있다.

    2. 콜백 지옥(CallBack Hell): 비동기식 I/O 처리를 위해 콜백 함수를 사용하는데, 중첩되는 콜백 함수 호출로 인해 코드의 가독성이 떨어질 수 있다. 


## 결론

- Node.js는 비동기식 처리 방식 사용, 빠른 속도, 빠른 개발, 대규모 데이터 처리, 웹 애플리케이션 개발 등을 할 때 효율성을 높일 수 있다.