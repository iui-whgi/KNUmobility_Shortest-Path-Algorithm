# KNUmobility_Shortest-Path-Algorithm
크누모빌리티 스터디에서 제작한 최단경로알고리즘 프로그램 소스파일입니다!


프로그램 실행파일 - 경북대학교ShortCut(github에는 exe파일이 따로 올라가지 않아 필요하시면 만들어야합니다)

설명:
코드에 보이는 것처럼 모든 건물(node)와 도로(edge)를 수작업으로 진행했기 때문에 부족한 부분이 있습니다. 따라서 실제 최단루트와는 다를 수 있습니다.
a*의 루트가 다익스트라와 다른 이유는 경로상 차이를 시각적으로 나타내기 위하여 임의로 휴리스틱 return 값에 *2를 해서 그렇습니다.
따라서 *2를 제거한다면 다익스트라와 동일한 결과가 나옵니다.  

requirements.txt 파일은 code열어서 개인적으로 실행해보실때 가상환경 파셔서 다운받으시면 됩니다.
