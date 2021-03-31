 

# Data Structures and Algorithms													

## 2021.1.18

- 数组、链表、跳表的原理和实现

- 三者的时间复杂度、空间复杂度

- 工程运用

- 跳表：升维思想 + 时间换空间

  | 数组    | 时间复杂度 |
  | ------- | ---------- |
  | prepend | O(1)       |
  | append  | O(1)       |
  | lookup  | O(1)       |
  | insert  | O(n)       |
  | delete  | O(n)       |

  | 链表    | 时间复杂度 |
  | ------- | ---------- |
  | prepend | O(1)       |
  | append  | O(1)       |
  | lookup  | O(n)       |
  | insert  | O(1)       |
  | delete  | O(1)       |

  跳表就是为了优化链表的

  假设索引有h级，最高级的索引有两个节点，n/(2^h) = 2，从而求得h = log2(n) - 1

  所以在跳表中查询任意数据的时间复杂度就是O(logn)

---

**练习步骤**

1.5-10分钟：读题与思考

2.有思路：自己开始做和写代码；不然，马上看题解

3.默写背诵、熟练

**初遇题目懵逼的时候怎么办?**

1. 思考能不能暴力解
2. 思考最简单的情况 
3. 思考最近的一次情况 倒推寻找重复性  最近重复子问题

---

**Leetcode**

283 移动零 双指针

11 盛水最多的容器  暴力法 ||  双指针

70 爬楼梯 斐波那契数列

---

## 2021.1.19

**LeetCode**  

1 两数之和          同理也是三种方法 

15 三数之和            1.暴力求解 三重循环 O(n ^ 3)		  2.hash表来记录 		3.排序加双指针

206 反转链表  	1.暴力法 2.递归

141 环形链表        1.暴力法 遍历链表 hash\set     2.快慢指针

26 删除排序数组中的重复项       快慢指针

189 旋转数组  1.使用临时数组 2.多次反转  

21 合并两个有序链表 1.暴力解法 2.递归解法

88 合并两个有序数组   1.合并后排序（忽略两个数组已经有序，时间复杂度差）2.双指针法 比较最小结果 存到新数组中 （占用额外空间）3.三指针法 从其中一个数组内部操作

---

## 2021.1.20

<img src="C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210120125109678.png" alt="image-20210120125109678" style="zoom:67%;" />

**栈和队列 *Stack & Queue***

- Stack :先入后出；添加、删除皆为O(1) 查询O(n)
- Queue:先入先出；添加、删除皆为O(1) 查询O(n)

**双端队列** ***Double-End Queue***   缩写 *deque*

- 添加、删除皆为O(1) 查询O(n)

**Priority Queue**

- 插入操作 O(1)
- 取出操作 O(logN)-按照元素的优先级取出（相对变慢了）
- 底层具体实现的数据结构较为多样和复杂：heap、bst、treap 

**leetcode**

20 有效的括号         *最近相关性  -> 栈问题  *先来后到 -> 队列问题 

- ​	暴力求解： 不断replace匹配的括号 替换为空string O(n^2)

- ​	Stack：将输入的左括号的匹配括号push入栈中，再按照顺序与后续的字符比较 最后判断栈是否为空

155 最小栈 使用两个栈 一个记录状态的变化 一个记录最小的元素

*84 柱状图中最大的矩形

- ​	暴力求解  O（n^2） 遍历每一个柱子 确定高度的前提下 寻找最大边界 求最大面积
- ​    Stack   遍历每一个柱子  满足条件 栈为空或者 元素大于等于栈顶元素时 压入栈 否则 计算上一个柱子的面积（因为左右边界已经确定 具体求stack和考虑offset）然后更新最大面积 

239 滑动窗口最大值  双端队列

---

## 2021.1.22

**四件套**

1.clarification

2.possible solutions --> optimal(time & space)

3.code

4.test cases

---

**leetcode**

242 有效的字母异位词

- 暴力,sort, sort_str 相等？ O(NlogN)
- hash,map -->统计每个字符的频次 维护一个长度为26 的哈希表 哈希函数 charAt(i) - 'a'

49 字母异位词分组 hash  * 字符串 排序的三个步骤 *map.getOrDefault() *map.values()

1.两数之和 hash a,b --> 

​	a + b == target --> for each a: check(target - a )exists in nums

---

Linked List 是特殊化的Tree	Tree是特殊化的图

树用递归很简洁

**二叉搜索树 Binary Search Tree**

二叉搜索树，也称有序二叉树(Ordered Binary Tree)、排序二叉树(Sorted Binary Tree)，是指一颗空树或者具有下列性质的二叉树：

1.左子树上 所有节点的值均小于它的根结点的值

2.右子树上所有结点的值均大于它的根结点的值

3.以此类推：左、右子树也分别为二叉查找树（**这就是重复性** **适合递归**）

**中序遍历是升序遍历**

查询和操作的时间复杂度都是O(logN)

**示例代码**

```java
public class TreeNode() {
    public int val;
    public TreeNode left, right;
    public TreeNode(int val) {
        this.val = val;
        this.left = null;
        this.right = null;
    }
} 
```

---

## 2021.1.23

**前中后序遍历代码摸板**

`preorder`

```python
def preorder(self, root):
    if root:
        self.traverse_path.append(root,val)
        self.preorder(root,left)
        self.preorder(root,right)
```

`inorder`

```python
def inorder(self, root):
    if root:
        self.inorder(root,left)
        self.traverse_path.append(root,val)
        self.inorder(root,right)
```

`postorder`

```python
def postorder(self, root):
    if root:
        self.postorder(root,left)
        self.postorder(root,right)
        self.traverse_path.append(root,val )
```

**leetcode**

94 二叉树的中序遍历  递归 迭代

144 二叉树的前序遍历 递归 迭代

145 二叉树的后序遍历 递归 迭代

---

## 2021.1.24

**递归 Recursion**  --> 通过函数体实现的循环

- 向下进入到不同递归层；向上又回到原来一层
- 通过参数实现函数不同层之间变量的传递
- 参数、全局变量改变其余的每一层都是新的

**代码摸板**

1. recursion terminator	递归终结条件
2. process logic in current level	处理当前层逻辑
3. drill down	下探到下一层
4. reverse the current level status if needed	清理当前层

```java
// Java
public void recur(int level, int param) {
    
// terminator
    if (level > MAX_LEVEL) {
        // process result
        return;
    }
    
    // process current logic
    process(level, param);
    
    // drill down
    recur( level: level + 1, newParam);
    
    // restore current status 
```

```python
# Python
def recursion(level, param1, param2, ...):
    
    # recursion terminator
    if level > MAX_LEVEL:
        # process_result 
        return 
    
    # process logic in current level
    process(level, data...) 
    
    # drill down    
    self.recursion(level + 1, p1, ...)  
    
    # reverse the current level status if needed
```

**注意事项**

1.不要进行人肉递归（最大误区）

2.找到最近最简方法，将其拆解成可重复解决的问题（找最近重复子问题）

3.数学归纳法思维

**leetcode**

22 括号生成

226 翻转二叉树

98 验证二叉搜索树 递归(BST-->中序遍历是递增的)

104 二叉树的最大深度 

111 二叉树的最小深度

## 2021.1.27

**分治 divid & conquer**

本质上就是递归 就是找最小重复性

**泛型代码摸板**

类比 递归

1. terminator
2. process(split your problem)
3. dirll down(subproblems), mertge (subresult)
4. reverse states

```java
Java
    private static int divide_conquer(Problem problem, ) {
    	//recursion terminator
        if (problem == NULL) {
            int res = process_last_result();
            return res;
        }
    
    	//prepare data
        subProblems = split_problem(problem)
        
        //conquer subproblems
        res0 = divide_conquer(subProblems[0])
        res1 = divide_conquer(subProblems[1]) 
        
        //process and generate the final result   *
        result = process_result(res0, res1);  
    
    	//revert the current level status
        return result;
}
```

**回溯 backtracking**

回溯是一种算法思想，可以用递归实现。

---

## 2021.1.31

**搜索遍历**

每个节点访问且仅访问一次

- 深度优先

- 广度优先

- 优先级有限 （启发式搜索）--> 深度学习 推荐算法

**示例代码**

```python
def dfs(node):
    if node in visited:
        #already visited
        return
    visited.add(node)
    # process current node
    # ...# logic here 
    dfs(node.left)
    dfs(node.right)
```

**深度优先 DFS Depth-First-Search**

recursion stack

```python
visited = set()
def dfs(node, visited):
    # if ndoe in visited :        terminator
    	# return				already visited
    visited.add(node)
    #process current node here.
    ...
    for next_node in node.children():
        if not next_ndoe in visited:
            dfs(next_node, visited)
```

**广度优先遍历 BFS Breadth-First-Search**

for loop queue 

```python
def BFS(graph, start, end):
    queue = []
    queue.append([start])
    visited.add(start)
    
    while queue:
        node = queue.pop()
        visited.add(node)
        
        process(node)
        nodes = generate_related_nodes(node)
        queue.push(nodes)
        
    # other processing work
    ...
```

## 2021.2.4

**贪心算法 greedy**

贪心算法是一种在每一步选择中都采取在当前状态下最好或最优（即最有利）的选择，从而希望导致结果是全据最好或最优的算法。



贪心算法与动态规划的不同在于它对每个子问题的解决方案都做出选择，**不能回退**。动态规划则会**保存以前的运算结果**，并根据以前的结果对当前进行选择，有回退功能。



- 贪心：当下做局部最优判断
- 回溯：能够回退
- 动态规划：最优判断 + 回退

贪心法可以解决一些最优化问题：图中最小生成树、求哈夫曼编码，然而对于生活中工程的问题，一般不能得到我们所要求的答案

一旦一个问题可以通过贪心法来解决，那么贪心法一般是解决这个问题的最好办法，由于贪心法的高效性以及其所求的的答案比较接近最优结果，贪心法也可以用作辅助算法或者直接解决一些要求结果不特别精确的问题。

**贪心算法适用的场景**

简单地说，问题能够分解成子问题来解决，子问题额最优解能递推到最终问题的最优解， 这种子问题最优解成为最优子结构。

## 2021.2.5

**二分查找**

*二分查找的前提*

1. 目标函数的单调性（单调递增或单调递减）
2. 存在上下界（bounded）
3. 能够通过索引访问 （index accessible）

**代码摸板**

```python
left, right = 0, len(array) - 1
while left <= right:
    mid = (left + rihgt) / 2
    if array[mid] == target:
        #find the target
        break or return result
	elif array[mid] < target:
        left = mid + 1
    else:
        right = mid -1
```

[Origin of Quake3&#39;s Fast InvSqrt()]: https://www.beyond3d.com/content/articles/8/

To most folks the following bit of C code, found in a few places in the recently released Quake3 source code, won't mean much. To the Beyond3D crowd it might ring a bell or two. It might even make some sense.

```c
float InvSqrt (float x){
  float xhalf = 0.5f*x;
  int i = *(int*)&x;
  i = 0x5f3759df - (i>>1);
  x = *(float*)&i;
  x = x*(1.5f - xhalf*x*x);
  return x;
}
```

**牛顿迭代法**

![[公式]](https://www.zhihu.com/equation?tex=x_%7Bn%2B1%7D+%3D+%5Cfrac%7B1%7D%7B2%7D%5Cleft%28+x_%7Bn%7D+%2B+%5Cfrac%7Ba%7D%7Bx_%7Bn%7D%7D%5Cright%29)

**二分法 防止溢出 技术处理**

- 一般写法

```java
int left = 0, right = n, mid;
mid = (left + right) / 2;
```

- 防止值移除写法

```java
int left = 0, right = n, mid;
mid =  left + (right - left) / 2;
```

- 位运算 更高效 需要注意的是位运算的优先级是低于加减运算的

```java
int left = 0, right = n, mid;
mid = left + (right - left >> 1)
```

## 2021.2.8

**动态规划 Dynamic Programming**

- Simplifying a complicated problem by breaking it down into simpler subproblems (in a recursive manner)
- Divide & Conquer + Optimal substructure 分治 + 最优子结构

**关键点**

动态规划和递归或者分治没有根本上的区别（关键看有无最优的子结构）

共性：找到重复子问题

差异性：最优子结构、中途可以淘汰次优解、储存中间状态

**即**

1. 子问题
2. 状态定义
3. dp方程

**递推公式（美其名曰：状态转移方程或者DP方程)**

​	Fib:

```java
opt[n] = opt[n - 1] + opt[n - 2]
```

​	二维路径：

```java
opt[i, j] = opt[i + 1][j] + opt[i][j + 1] //且判断 opt[i][j]是否为空地
```

字符串 最长公共子序列  使用  **二维数组**  行和列代表两个字符串的子序列

1. 先进行初始化 
   1. 1	最后一个字符相同 上一个的两个子序列的最大值 + 1
   2. 1	上一个的两个子序列的最大值

![image-20210216113617923](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210216113617923.png)

**小结**

1. 打破自己的思维惯性，形成机器思维
2. 理解复杂逻辑的关键
3. 也是职业进阶的要点要领

## 2021.2.17

[一句话团灭股票问题]: https://github.com/labuladong/fucking-algorithm/blob/master/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%B3%BB%E5%88%97/%E5%9B%A2%E7%81%AD%E8%82%A1%E7%A5%A8%E9%97%AE%E9%A2%98.md

股票问题的 base case 及 状态转移方程

```java
base case：
dp[-1][k][0] = dp[i][0][0] = 0
dp[-1][k][1] = dp[i][0][1] = -infinity

状态转移方程：
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
```

---

**字典树 trie**

- 数据结构

字典树，即Trie树，又称单词查找树或键树，是一种树形结构。典型应用就是用于统计和排序大量的字符串（但不仅限于字符串)，所以经常被搜索引擎系统用于文本词频统计。

他的优点是：最大限度地减少无谓的字符串比较，查询效率比哈希表高。

- 核心思想

Trie树的核心思想是空间换时间

利用字符串的公共前缀来降低查询时间的开下以达到提高效率的目的

- 基本性质

节点本身不存完整单词；

从根节点到某一结点，路径上经过的字符连接起来，为该节点对应的字符串；

每个节点的所有子节点路径代表的字符都不相同

- 节点的内部实现

![image-20210217215209593](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210217215209593.png)

## 2021.2.18

**并查集  Disjoint Set**

**适用场景**

- 组团、配对问题
- Group or not?

**基本操作**

- makeSet(s): 建立一个新的并查集，其中包含s个单元素集合
- unionSet(x, y): 把元素x和元素y所在的集合合并，要求x和y所在的集合不相交，如果相交则不合并
- find(x):找到元素x所在的集合的代表，该操作也可以用于判断两个元素是否位于同一个集合，只要将它们各自的代表比较一下就可以了

**java实现**

```java
class UnionFind {
    private int count = 0;
    private int[] parent;
    public UnionFind(int n) {
        count = n;
        parent = new int[n];
        for (int i = 0; i < n; i++) {
			parent[i] = i;
        }
    }
    public int find(int p) {
        while (p != parent[p]) {
            parent[p] = parent[parent[p]];
            p = parent[p];
        }
        return p;
    }
    public void union (int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        if (rootP == rootQ) return;
        parent[rooP] = rootQ;
        count --;
    }
}
```

## 2021.2.20

**初级搜索**

1. 朴素搜索
2. 优化方式：不重复（fibonacci）、剪枝（生成括号问题）
3. 搜索方向：
   	DFS BFS

**高级搜索**

剪枝

双向搜索

启发式搜索  Heuristic Search (A*)

A* search 

```python
def AstarSearch(graph, start, end):
    pq = collection.priority_queue() #优先级 -> 估价函数
    pq.append([start])
    visited.add(start)
    
    while pq:
        node = pq.pop() #can we add more intelligence here
        visited.add(node)
        process(node)
        nodes = generate_related_nodes(node)
        unvisited = [node for node in nodes if node not in visited]
        pq.push(unvisited)
```

**估价函数**

启发式函数：h(n), 它用来评价哪些结点最有希望的是一个我们要找的结点，h(n)会返回一个非负实数，也可以认为是从结点n的目标结点路径的估计成本

启发式函数是一种告知搜索方向的方法，它提供了一种明智的方法来猜测哪个邻居结点会导向一个目标

## 2021.2.21

**平衡二叉树** self-balancing(or height-balanced) binary search tree

**Implementations**

- 2-3 tree *
- AA tree
- AVL tree *
- B-tree *
- Red-black tree *
- Scapegoat  tree
- Splay tree
- Tango tree
- Treap
- Weight-balanced tree

二叉搜索树的查询效率只与它的高度有关，与结点个数无关

---

**AVL tree** 自平衡二叉搜索树

Balance Factor(平衡因子) :是它的左子树的高度减去它的右子树的高度（有时相反）

balance factor = {-1, 0, 1}

通过旋转操作来进行平衡（四种）

**旋转操作**

1.左旋 ->右右子树

![image-20210221151359179](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151359179.png)<img src="C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151412324.png" alt="image-20210221151412324" style="zoom:125%;" />

2.右旋 -> 左左子树

![image-20210221151503994](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151503994.png)<img src="C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151520409.png" alt="image-20210221151520409" style="zoom:130%;" />

3.左右旋 -> 左右子树

![image-20210221151546309](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151546309.png)![image-20210221151553098](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151553098.png)<img src="C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151601178.png" alt="image-20210221151601178" style="zoom:150%;" />

4.右左旋 -> 右左子树

![image-20210221151609560](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151609560.png)![image-20210221151615830](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151615830.png)<img src="C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221151621968.png" alt="image-20210221151621968" style="zoom:150%;" />

不足：结点需要存储额外信息，且调整次数频繁

下面引入近似平衡二叉树

---

**红黑树 Red-black tree**

- 红黑树是一种**近似平衡**的二叉搜索树(Binary Search Tree)，它能够确保任何一个结点的左右子树的**高度差小于两倍**。具体来说，红黑树是满足如下条件的二叉搜索树：
  每个结点要么是红色、要么是黑色
- 根节点是黑色
- 每个叶结点（NIL结点，空结点）是黑色的
- 不能有相邻接的两个红色结点
- 从任意结点到其每个叶子的所有路径都包含相同数目的黑色结点 

**关键性质**

从根到叶子的最长路径不多于最短的可能路径的两倍长

**对比 (important) **

- AVL trees provide **faster lookups** than Red Black Trees because they are **more strictly balanced**
- Red Black Trees provide **faster insertion and removal** operations than AVL trees as fewer rotations are done due to relatively relaxed balancing
- AVL trees store balance **factors or height** with each node, thus requires storage for an integer per node whereas Red Black Tree require only 1 bit of information per node.
- Red Black Trees are used in most of the **language libraires** **like map. multimap, multisetin C++** whereas AVL trees are used in **databases** where faster retrievals are required.

![image-20210221153239726](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221153239726.png)

---

**位运算**

为什么需要位运算

- 机器里的数字表示方式和存储格式都是 二进制
- 十进制 <-> 二进制 如何转换？

位运算符

![image-20210221154740809](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221154740809.png)

![image-20210221162623642](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221162623642.png)

XOR - 异或

异或：相同为0，不同为1，也可用 "不进位加法"来理解。

异或操作的一些特点：

![image-20210221161151898](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221161151898.png)

指定位置的位运算

![image-20210221161750104](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210221161750104.png)

实战位运算要点

- 判断奇偶

   x % 2 == 1 --> (x & 1) == 1

   x % 2 == 0 --> (x & 1) == 0

-  x >> 1 -- > x / 2

   即： x = x / 2; --> x = x >> 1

  ​		mid = (left + right) / 2; --> mid = (left + right) >> 1;

- X = X & (X - 1) 清零最低位的1

- X & -X => 得到最低位的1

- X & ~X => 0

## 2021.2.22

**布隆过滤器 bloom filter**

一个很长的<u>二进制</u>向量和一系列<u>随机映射函数</u>。布隆过滤器可以用于检索一个元素是否在一个集合中。

有点是空间效率和查询时间都远远超过一般的算法

缺点是有一定的误识别率和删除困难

- 如果查 不存在 则 **一定** 不存在，如果查到 存在 则 **不一定** 存在 

**应用**

1. 比特币网络
2. 分布式系统(Map-Reduce) -Hadoop \ Search Engine
3. Redis 缓存
4. 垃圾邮件、评论等的过滤

---

**LRU Cache**

- 两个要素：大小、替换策略
- Hash Table + Double LinkedList
- O(1)查询 O(1) 修改、更新

**工作示例**

![image-20210222134527860](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210222134527860.png)

替换策略

LFU -least frequently used

LRU least recently used

---

**排序算法**

1. 比较类排序:
   通过比较来决定元素间的相对次序，由于其时间复杂度不能突破O(nlogn)，因此也被称为非线性时间比较类排序
2. 非比较类排序：
   不通过比较来决定元素间的相对次序，它可以突破基于比较排序的时间下界，以线性时间运行，因此也成为线性时间非比较类排序

![image-20210222173218593](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210222173218593.png)

![Inkedimage-20210222173240636_LI](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\Inkedimage-20210222173240636_LI.jpg)

**初级排序 - O(n ^ 2)**

1.选择排序(Selection Sort)

​	每次找最小值，然后放到待排数组的起始位置

2.插入排序(Insertion Sort)

​	从前到后逐步构建有序序列；对于对排序数组，在已排序序列中从后向前，找到相应位置并插入

3.冒泡排序(Bubble Sort)

​	嵌套循环，每次查看相邻的元素如果逆序，则交换

---

**高级排序 -O(N * LogN)**

快速排序 (quick sort) -分治

数组取标杆pivot，将小元素放pivot左边，大元素放右侧，然后依次对右边和右边的子数组继续快排；以达到整个序列有序.

```java
public static void quickSort(int[] array, int begin, int end) {
        if (end <= right) return;
        int pivot = partition(array, begin, end);
        quickSort(array, begin, pivot - 1);
        quickSort(array, pivot + 1, end);
    }

    static int partition(int[] a, int begin, int end) {
        //pivot:标杆位置, counter:小于pivot的元素的个数
        int pivot =  end, counter = begin;
        for (int i = begin; i < end; i++) {
            if (a[i] < a[pivot]) {
                //swap a[i] & a[counter]
                int temp = a[counter]; a[counter] = a[i]; a[i] = temp;
                counter++;
            }
        }
        //swap a[pivot] & a[counter]
        int temp = a[pivot]; a[pivot] = a[counter]; a[counter] = temp;
        return counter;
    }
```

归并排序(Merge Sort) -分治       (模糊理解为快排的逆序)

1.把长度为n大的输入序列分为两个长度为n/2的子序列；

2.对这两个子序列分别采用归并排序；

3.将两个排序好的子序列合并成一个最终的排序序列。

```java
public static void mergeSort(int[] array, int left, int right) {
	if (right <= left) return;
    int mid = (left + right) >> 1;
    mergeSort(array, left, mid);
    mergeSort(array, mid + 1, right);
    merge(array, left, mid, right);
}
static void merge(int[] arr, int left, int mid, int right) {
    int[] temp = new int [right - left + 1];
   	int i = left, j = mid + 1, k = 0;
    while (i <= mid && j <= right) temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    System.arraycopy(temp, 0, arr, left, temp.length);
}
```

归并和快排具有相似性，但步骤顺序相反

归并：先排序左右子数组，然后合并两个有序子数组

快排：先调配出左右子数组，然后对于左右子树组进行排序

---

堆排序 (Heap Sort) --堆插入 O(logN),取最大/小值 O(1)

1.数组元素依次建立小顶堆

2.依次取堆顶元素，并删除

**使用priorityQueue 实现堆排序**

```java
public static void heapSort(int[] array) {
    PriorityQueue<Integer> queue = new PriorityQueue<>();
    for (int j : array) {
        queue.offer(j);
    }
    for (int i = 0; i < array.length; i++) {
        array[i] = queue.remove();
    }
}
```

**使用数组维护一个大顶堆 实现堆排序**

```java
public static void heapSort(int[] array) {
        if (array.length == 0) return;
        int length = array.length;
        for (int i = length / 2 - 1; i >= 0; i--) {
            heapify(array, length, i);
        }
        for (int i = length - 1; i >= 0; i--) {
            int temp = array[0]; array[0] = array[i]; array[i] = temp;
            heapify(array, i, 0);
        }
    }

    static void heapify(int[] array, int length, int i) {
        int left = 2 * i + 1,  right = 2 * i + 2;
        int largest = i;
        if (left < length && array[left] > array[largest]) {
            largest = left;
        }
        if (right < length && array[right] > array[largest]) {
            largest = right;
        }
        if (largest != i) {
            int temp = array[i]; array[i] = array[largest]; array[largest] = temp;
            heapify(array, length, largest);
        }
    }
```

---

**特殊排序 -O(N)**

1计数排序(Counting Sort).

​	计数排序要求输入的数据必须是有确定范围的整数，将输入的数据值转化为键存储在额外开辟的数组空间中；然后依次把计数大于1的填充回原数组

2.桶排序(Bucket Sort) 

​	桶排序的工作原理：假设输入数据服从均匀分布，将数据分到有限数量的桶里，每个桶再分别排序（有可能再使用别的排序算法或是以递归方式继续使用桶排序进行排序）

3.基数排序(Radix Sort)

​	计数排序是按照低位先排序，然后收集；再按照高位排序，然后再收集；依次类推，直到最高位。有时候有些属性是有优先级顺序的，先按低优先级排序，再按高优先级排序 

## 2021.2.26

**高级动态规划**

![image-20210226182143502](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210226182143502.png)

**字符串匹配算法**

brute force O(mn)

```java
public static int forceSearch(String txt, String pat) {
    int M = txt.length();
    int N = pat.length();
    for (int i = 0; i <= M - N; i++) {
        int j;
        for (j = 0; j < N; j++) {
            if (txt.charAt(i + j) != pat.charAt(j)) break;
        }
        if (j == N) return i;
    }
    return - 1;
}
```

**Rabin-Karp算法**

在朴素算法的基础上加以优化，为了避免挨个字符堆目标字符串和字串进行比较，我们可以尝试一次性判断两者是否相等。因此，我们需要一个好的哈希函数(hash function)。通过哈希函数，我们可以算出字串的哈希值，然后将它和目标字符串中的字串的哈希值进行比较。

算法思想：

1.假设字串的长度为M(pat),目标字符串的长度为N(pat)

2.计算字串的hash值hash pat

3.计算目标字符串txt中每个长度为M的字串的hash值（共需要计算 N - M + 1次)

4.比较hash值：如果hash值不同，字符串必然不匹配；如果hash相同，还需要用朴素算法再次判断

```java
public final static int D = 256;
public final static int Q = 9997;
static int RabinKarpSerach(String txt, String pat) {
    int M = pat.length(); 
    int N = txt.length();   
    int i, j;  
    int patHash = 0, txtHash = 0;
    for (i = 0; i < M; i++) {  
        patHash = (D * patHash + pat.charAt(i)) % Q;
        txtHash = (D * txtHash + txt.charAt(i)) % Q; 
    }  
    int highestPow = 1; // pow(256, M-1) 
    for (i = 0; i < M - 1; i++)     
        highestPow = (highestPow * D) % Q; 
    for (i = 0; i <= N - M; i++) { // 枚举起点  
        if (patHash == txtHash) {  
            for (j = 0; j < M; j++) {  
                if (txt.charAt(i + j) != pat.charAt(j))  
                    break;           
            }      
            if (j == M)    
                return i; 
        }       
        if (i < N - M) {   
            txtHash = (D * (txtHash - txt.charAt(i) * highestPow) + txt.charAt(i + M)) % Q; 
            if (txtHash < 0)              
                txtHash += Q;     
        } 
    }    
    return -1;
}
```

**KMP算法**

1.prefix table 最长公共前后缀

2.匹配成功右移一位，匹配失败 根据对应的prefix table 将 该位置移到匹配失败的位置，

![image-20210306141457132](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210306141457132.png)

![image-20210306141844340](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210306141844340.png)

![image-20210306143305306](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210306143305306.png)