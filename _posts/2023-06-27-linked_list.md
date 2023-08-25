---
title: Linked List (링크드 리스트)
date: 2023-06-27 00:21:00 +0900
categories: [Data_Structure]
tags: [Data_Structure]
---

![](https://velog.velcdn.com/images/acadias12/post/1c906350-6bb8-4f64-b431-0a98d0941ce0/image.png)

C언어를 기반으로 자료구조에서의 Linked List에 대해 정리해보려한다.

## Linked List란?
>링크드 리스트(linked list)는 데이터 요소들을 순서대로 저장하기 위한 자료 구조이다. 각각의 데이터 요소는 노드(node)라고 불리는 객체로 표현되며. 각 노드는 데이터와 다음 노드를 가리키는 포인터(링크)로 구성된다.

---

## Linked List 예시
![](https://velog.velcdn.com/images/acadias12/post/3836f2a4-a512-44fe-b42c-69aafd969f8f/image.png)

> Head: 링크드 리스트의 첫 번째 노드를 가르키는 포인터 

![](https://velog.velcdn.com/images/acadias12/post/0d12e594-eea2-4bd0-b63b-270a14be87ab/image.png)

>위의 사진처럼 Head 포인터를 NULL값으로 설정하면 노드의 삽입과 삭제 관리를 더욱 편리하게 할 수있다.

---

## Linked List 기능 구현

>
void InitList(LinkedList* plist) : 리스트를 초기화 시킴
bool IsEmpty(LinkedList* plist) : 리스트가 비어있는지 확인 함 
void InsertMiddle(LinkedList* plist, int pos, Data item) : k번째의 위치에 요소를 삽입 함
void RemoveMiddle(LinkedList* plist, int pos) : k번째의 위치의 요소를 삭제함
Data ReadItem(LinkedList* plist, int pos) : k번째의 위치의 요소를 가져옴
void PrintList(LinkedList* plist) : 리스트의 각 요소들을 출력함
void ClearList(LinkedList* plist) : 리스트의 모든 노드를 삭제함

---

## Linked List 기능 코드

#### Linked List 구현 코드

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int data;
typedef struct _node{
	data item;
	struct _node* next;
}node;
typedef struct {
	node* head;
	int len;
}linkedlist;
```

#### InitList 구현 코드

```cpp
void initlist(linkedlist* plist) {
	plist->head = (node*)malloc(sizeof(node));
	plist->head->next = NULL;
	plist->len = 0;
}
```

#### IsEmpty 구현 코드

```cpp
bool isempty(linkedlist* plist) {
	return plist->len == 0;
}
```

#### InsertMiddle 구현 코드

```cpp
void insertmiddle(linkedlist* plist, int pos, data item) {
	node* cur, *newnode;
	if (pos<0 || pos > plist->len) exit(1);
	newnode = (node*)malloc(sizeof(node));
	newnode->item = item;
	newnode->next = NULL;
	cur = plist->head;
	for (int i = 0; i < pos; i++) {
		cur = cur->next;
	}
	newnode->next = cur->next;
	cur->next = newnode;
	plist->len++;
}
```

#### RemoveMiddle 구현 코드

```cpp
void removemiddle(linkedlist* plist, int pos) {
	node* cur,*temp;
	if (pos<0 || pos>plist->len) exit(1);
	cur = plist->head;
	for (int i = 0; i < pos; i++) {
		cur = cur->next;
	}
	temp = cur->next;
	cur->next = cur->next->next;
	plist->len--;
	free(temp);
	
}
```

#### ReadItem 구현 코드

```cpp
data readitem(linkedlist* plist, int pos) {
	node* cur;
	if (pos<0 || pos > plist->len) exit(1);
	cur = plist->head->next;
	for (int i = 0; i < pos; i++) {
		cur = cur->next;
	}
	return cur->item;
}
```

#### PrintList 구현 코드

```cpp
void printlist(linkedlist* plist) {
	for (node* cur = plist->head->next; cur != NULL; cur = cur->next) {
		printf("%d ", cur->item);
	}
}
```

#### ClearList 구현 코드

```cpp
void clearlist(linkedlist* plist) {
	while (plist->head->next != NULL) {
		removefirst(plist);
	}
	free(plist->head);
}
```

---

## Linked List의 시간 복잡도

| Queue | 탐색 | 삽입 | 삭제 |
| :- | - | :-: |:-:|
| Big-O | O(n) | O(1) | O(1) |

---

## 느낀점 

Linked List를 배워보면서 Array List와 비교해 보았을 때 Linked List는 삽입과 삭제의 부분에서 상수시간에 삽입과 삭제가 가능하다는 점에서 이점을 가진다는 것을 알게 되었다. 이번 포스트에서 다룬 Linked List는 단일 방향만 가지는 Single Linked List이다. Double Linked List에 대해서도 공부해 봐야겠다.