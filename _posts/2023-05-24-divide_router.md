---
title: Node.js 라우터 분리
date: 2023-05-24 12:43:00 +0900
categories: [Node.js]
tags: [Node.js]
---

![](https://velog.velcdn.com/images/acadias12/post/7cd9693d-61b7-4f55-b255-9f6f47f5c3a6/image.png)

Node.js를 공부하다가 어려웠던 개념인 라우터 분리에 대해 정리해보려한다.

## 라우터(Router)란?
> 라우터는 웹 애플리케이션에서 클라이언트의 요청을 처리하고 적절한 핸들러 함수로 라우팅하는 역할을 담당하는 기능이다.

라우터는 URL의 경로와 HTTP 메소드(GET, POST 등)를 기반으로 요청을 분류하고 처리할 핸들러 함수로 라우팅한다. 

## 라우터(Router)를 분리하는 이유

> 라우터를 분리함으로써 코드의 구조화, 모듈화, 재사용성, 유연성, 테스트 용이성 등 다양한 이점을 얻을 수 있다.

코드의 규모가 커질수록 라우터를 분리하는 것은 필수적이라 생각한다.

코드로 예시를 들어보면

**라우터 분리를 하지 않은 메인코드**

``` javascript
const express = require('express');
const app = express();

app.get('/user',function(req,res){
    res.send("Hello user");
  });

  app.get('/user/add',function(req,res){
    res.send("Hello user add");
  });

  app.get('/user/delte',function(req,res){
    res.send("Hello user delete");
  });

  app.get('/main',function(req,res){
    res.send("Hello main");
  });

  app.get('/main/login',function(req,res){
    res.send("Hello main login");
  });

  app.get('/main/logout',function(req,res){
    res.send("Hello main logout");
  });
```

**라우터를 분리한 후 메인 코드**
```javascript
const express = require('express');
const app = express();

const main = require('./routes/main'); // main 라우터
app.use('/main',main);

const user = require('./routes/user'); // user 라우터
app.use('/user',user);


app.listen(3030,function(){
    console.log('Connected 3030port!');
  });
```

라우터 분리를 한 후 메인 코드가 매우 간결해지는 것을 확인할 수 있다.


## 라우터 분리를 하는 방법

위의 라우터 분리를 하지 않은 코드를 보면 user와 main이 중복되는 것을 볼 수 있다. 따라서 user는 user로, main은 main으로 분리해주면 된다.

**main라우터 코드**

```javascript
const express = require('express');
const router = express.Router();
router.get('/',function(req,res){
    res.send("Hello main");
  });

router.get('/login',function(req,res){
    res.send("Hello main login");
  });

router.get('logout',function(req,res){
    res.send("Hello main logout");
  });

module.exports = router; 
```

**user라우터 코드**

```javascript
const express = require('express');
const router = express.Router();

router.get('/',function(req,res){
    res.send("Hello user");
  });

router.get('/add',function(req,res){
    res.send("Hello user add");
  });

router.get('/delte',function(req,res){
    res.send("Hello user delete");
  });

module.exports = router;
```
**메인함수 코드**

```javascript
const express = require('express');
const app = express();

const main = require('./routes/main'); // main 라우터
app.use('/main',main);

const user = require('./routes/user'); // user 라우터
app.use('/user',user);


app.listen(3030,function(){
    console.log('Connected 3030port!');
```

이렇게 라우터를 분리해줌으로써 코드의 구조화, 모듈화, 재사용성, 유연성, 테스트 용이성 등 다양한 이점을 얻을 수 있다.

## 느낀점

라우터 분리하는 것을 배워보면서, 라우터가 만약 상당히 많다면 라우터를 분리하더라도 메인 함수의 코드가 매우 길어질 것이라 생각했다. 따라서, 나는 라우터가 많더라도 메인 함수가 길어지지 않는 방법에 대해 고민을 해봤다. 만약 라우터를 클래스로 반환한 뒤 배열로 넘겨준다면 이 문제를 해결할 수 있지 않을까? 라는 생각을 하였고, 이에 대해 더 공부해 봐야겠다고 생각했다.