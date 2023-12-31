---
title: 깃허브 블로그 포스팅하기
date: 2023-04-30 20:40:00 +0900
categories: [Github]
tags: [Github]
---


# 깃허브 블로그 포스팅하기

안녕하세요 여러분! 이번 포스팅에서는 깃허브 블로그에 포스팅하는법을 알아보겠습니다~! 😁
<br><br/>


## 마크다운 (MARKDOWN)
깃허브 블로그를 포스팅할 때는 마크다운 언어를 사용해 포스팅합니다!
<br><br/>  

* 마크다운이란?
	>마크다운 언어는 문서를 작성할 때 일반 텍스트로 작성하면서도 일부 태그를 사용하여 문서를 서식화하는 경량 마크업 언어입니다. 마크다운은 쉽게 읽을 수 있으며, HTML로 변환이 간단하다는 장점이 있습니다.  


* 마크다운 언어 문법
	* 제목을 입력할 때 
		>제목을 입력할 때앞에 #을 붙입니다. #은 총 6개까지 붙일 수 있으며 글자 크기를 조절할 수 있습니다. 
	
		> # h1 -> #h1
		> ## h2 -> ##h2
		> ### h3 -> ###h3 
	
	* 줄바꿈
		> 줄바꿈을 할 때는 1. 빈 줄을 추가 2. 스페이스바를 두번 입력 3. br 태그를 사용을 통해 줄바꿈을 할 수 있습니다.

			ex)     첫번째문장
					두번째 문장
					
					첫번째 문장  두번째문장
					
					첫번째 문장<br>두번째문장

	* 가로줄 추가 
		> 가로줄을 추가할 때는 --- 혹은 *** 중 하나를 선택해서 사용하시면 됩니다.
		ß

				ex)
				---
				가로줄
				***

	* 글머리 기호 추가
		> 글머리 기호를 추가하려면 * + - 중 하나를 택해서 사용하거나, 혼합해서 사용하면 글머리 기호를 추가 할 수 있습니다. 
		**TAB키를 사용해서 들여쓰기를 하면 여러 단계를 가진 목록을 만들 수 있습니다!**
			
* 예시
	* 예시
		

			ex) 
			* 글머리
				* 글머리

	* 텍스트 강조
		> 텍스트의 일부분을 굵게 또는 기울임체를 써서 강조할 수 있습니다.
		
			ex)	굵게: ** 혹은 __로 감싸기
					기울임체: * 혹은 _로 감싸기
					굵은 기울임체: ***혹은 ___로 감싸기
					취소선: ~~로 감싸기


		> **굵게** <br>
			*기울임체* <br>
			***굵은 기울임체*** <br>
			~~취소선~~
	* 인용문 추가
		> 인용문을 추가하려면 >를 사용하면 됩니다! 인용문 내의 인용문은 >>를 사용하여 나타낼 수 있습니다.
		
			ex) >인용문
					>> 인용문 내의 인용문

		>인용문
		>>인용문 내의 인용문

	* 소스코드 추가
		> 블로그를 작성할 때 코드를 삽입하려면 grave기호를(`) 사용하시면 됩니다. ```뒤에 python, javascript 등의 언어를 지정해주면 해당 언어에 맞는 소스코드로 바뀝니다!

			ex) `소스코드`
					```여러줄의 소스코드```
		
		```python
		def function():
			print("hello")
		```	

	<br><br/>
	마크다운 언어 문법은 여기까지 알아보도록 하고 블로그 포스팅하는 법에 대해 알아보겠습니다!! 😁😁

## 블로그 포스팅
<br><br/>
>  깃허브 블로그에 글을 포스팅 하려면 마크다운 언어로 작성된 파일을 _posts폴더에 넣어주어야 합니다! 

<img width="1456" alt="스크린샷 2023-04-30 오후 8 19 33" src="https://user-images.githubusercontent.com/127467808/235350287-3503594b-a7b4-45cc-93e1-418824f655ed.png">

> 오른쪽 상단에 Create new file을 눌러 새 파일을 만들어줍니다!

<img width="911" alt="스크린샷 2023-04-30 오후 8 21 26" src="https://user-images.githubusercontent.com/127467808/235350292-18375353-4ffa-4a42-bba1-60405084943e.png">

> 여기서 가장 중요한 것은 파일명을 __YYYY-MM-DD-TITLE.md__ 형식으로 작성을 해줘야 합니다! 이후 마크다운 언어를 사용하여 내용을 적고 커밋해주면 됩니다!!

커밋 후 1~2분 후 블로그에 글이 올라간 것을 확인 하실 수 있습니다!😆😆

오늘은 블로그 포스팅하는 법에 대해 알아보았습니다. 블로그 포스팅을 통해 여러분이 원하는 블로그를 만들어 나가면 좋을거 같습니다!

깃허브 블로그 포스팅을 하느라 수고하신 여러분!! 고생 많으셨고 행복한 하루 보내세요!!😎😎
				
