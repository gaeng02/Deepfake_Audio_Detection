# 2024 July 08

### Issue #1 :: train data의 크기가 다르다.

각자의 방식대로 진행해볼 예정

- 모든 data의 앞부분, 뒷부분에 아무 데이터가 없는 부분을 잘라냄 (→ 또 다른 data가 생성). 이를 통해 적절히 padding 혹은 cut 예정

→ 오히려 deep-learning 코드는 간결하지만, pre-processing 과정이 복잡함.


### Issue #2 :: test_set을 어떻게 준비하고 있는지

한명이 코드 작성하면 공유하는 방법으로


### Issue #3 :: Over-fitting 문제에 대해 어떻게 생각하는가

test_set은 train_set과 동일하게 deep-fake 과정을 거친 후 합쳐진 오디오 파일이기 때문에, train_set에 과적합이 문제가 될 것 같지 않다. <br>

따라서, train_set의 pre-processing과 test_set의 데이터 처리 과정이 복잡할 것으로 예상됨.