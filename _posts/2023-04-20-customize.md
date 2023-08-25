---
title: 깃허브 블로그 꾸미기
date: 2023-04-20 15:50:00 +0900
categories: [Github]
tags: [Github]
---

# 깃허브 블로그 꾸며보기
안녕하세요 여러분! 이번 포스팅은 깃허브 블로그를 꾸미는 방법에 대해 알아봅시다!

## Avatar(아바타) 바꾸기


![avatar](https://user-images.githubusercontent.com/127467808/233265090-747465f9-6aa4-41d4-8d76-b7f94241c6eb.png)
> 위에 표시한 부분을 다른 사진으로 교체하기 위해서 자신의 깃허브 블로그 repository에 들어가 assets폴더 클릭 -> img/favicons폴더 안에 자신의 avatar사진을 Add file을 통해 넣어줍니다!

![avtar2](https://user-images.githubusercontent.com/127467808/233266251-30b14284-5185-4af2-8289-716285ef7ed1.png)
 저는 profile2라는 png 파일을 넣었습니다! 😀 

![avatar3](https://user-images.githubusercontent.com/127467808/233266817-2facdd16-f369-4435-8306-2b19b860ea1d.png)
> 이미지 파일을 넣었다면 repository에서 _config.yml파일에 들어갑니다! _config.yml파일을 내리다 보면 위 사진처럼 avatar를 변경할 수 있는 부분이 있습니다. Avatar를 바꾸기 위해 1번 부분을 " "만 남기고 지워주고, 2번 부분에 위에서 했던 이미지 파일 경로를 넣어줍니다!
> ex) /assets/img/favicons/profile2.png  경로를 틀리게 입력하시면 안됩니다!!😰
![avatar5](https://user-images.githubusercontent.com/127467808/233268509-26e3e554-874d-4de4-8da0-ee64d0326a99.png)

자 이제 자신의 깃허브 블로그를 들어가 Avatar가 바뀌었는지 확인 해봅시다! 😃

![avatar4](https://user-images.githubusercontent.com/127467808/233268731-0446c21e-04d8-4f29-87e7-8d3a7d042882.png)
귀여운 돼지 그림으로 잘 바뀌었네요!!😙

##  테마 모드 바꾸기 (ex 다크 모드)
깃허브 블로그에 테마 모드도 변경할 수 있습니다! 한번 바꿔 볼까요? 🤭 

![mode](https://user-images.githubusercontent.com/127467808/233269580-20db2297-700c-49a5-b534-56afbdf729fa.png)
> 깃허브 블로그 repository에 _config.yml에 들어가 위 사진처럼 theme_mode를 찾아봅니다! theme_mode를 보면 light모드와 dark모드가 있는 것을 확인해 볼 수 있는데요 기본적으로는 light 모드가 사용됩니다!
> 
![mode2](https://user-images.githubusercontent.com/127467808/233270107-f7d15d61-8a7e-4af0-b709-85c530adb514.png)
> 위 사진처럼 변경해주면 theme_mode를 다크 모드로 변경할 수 있습니다!

다크 모드로 잘 적용되었는지 확인해볼까요?😄
![mode3](https://user-images.githubusercontent.com/127467808/233270359-317d2056-f222-4691-a36f-aa07596b295d.png)
> 다크 모드가 잘 적용된 것을 확인할 수 있습니다!

## Favicon 바꾸기

![favicon](https://user-images.githubusercontent.com/127467808/233271133-26586a57-2222-455b-bcdf-8c04a02653fc.png)
Favicon은  위 사진을 바꾸는 건데요! 한번 바꿔봅시다! 😃

>[**Real Favicon Generator**](https://realfavicongenerator.net/) -> 원하는 이미지를 favicon으로 만들어주는 유용한 사이트입니다! 원하는 이미지를 favicon으로 변경해 Favicon package를 다운을 받아줍니다!

![favicon3](https://user-images.githubusercontent.com/127467808/233273701-85ba407e-2167-479a-9ec8-bf2cd56524ac.png)
> Favicon package를 다운받으면 이런 파일들이 보입니다! 위 사진에 밑줄 친 파일은 모두 삭제해 주면 됩니다!

![favicon4](https://user-images.githubusercontent.com/127467808/233276495-fd469776-7e8d-4b32-a1bc-1f223a4798b3.png)
> 압축된 파일을 풀고 깃허브 블로그 repository에 assets->img/favicons 안에 위 사진처럼 파일을 덮어쓰기를 해줍니다!

![f5](https://user-images.githubusercontent.com/127467808/233277274-f0f6b1b4-2751-4f3a-b15a-99359ef32619.png)
![f6](https://user-images.githubusercontent.com/127467808/233277277-043ee0e1-a9ab-4060-8b22-d1901cf9461c.png)
> Favicon 이미지를 넣어줬다면 repository에 들어가 -> _includes 폴더 -> favicons.html을 들어가줍니다!

![f7](https://user-images.githubusercontent.com/127467808/233280213-d2f1ced8-8347-4336-a604-0ff669e1e69a.png)
![f8](https://user-images.githubusercontent.com/127467808/233280229-05dc01fd-2b7b-429c-8560-736595e6d37a.png)
 > assets/img/favicons에 덮어쓰기 한 파일에 이름이나 사이즈가 달라질 수 있기 때문에 favicons.html에서 변경사항을 추가해줍니다!

위 과정을 마치셨다면 Favicon 변경이 완료됩니다! 한번 확인 해봅시다!😁
![f9](https://user-images.githubusercontent.com/127467808/233280245-ddbe2f32-d8ec-428e-99b1-719548ee3fae.png)
>Favicon이 잘 바뀐 것을 확인할 수 있습니다!!

여러분 고생하셨습니다! 지금까지 블로그를 만들어보고, 블로그를 꾸미기까지 헤보았는데요 다음 포스팅에서는 블로그에 글 쓰는 법(포스팅)에 대해 포스팅을 해보겠습니다. 감사합니다!!

🫡블로그를 꾸미느라 수고하신 여러분! 즐거운 하루 보내세요!! 🫡