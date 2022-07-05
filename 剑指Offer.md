# :star:[剑指Offer](https://leetcode-cn.com/problemset/lcof/) 

> `f27E` 仅以纪念学习路上的你我

## :apple:数据结构 11

### [05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

```java
class Solution {
    public String replaceSpace(String s) {
       StringBuilder x = new StringBuilder();
       for (char c : s.toCharArray()) {
           if (c == ' ') x.append("%20");
           else x.append(c);
       }
       return x.toString();
    } 
}
```
### [06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

`recursion`

```java
class Solution {
    ArrayList<Integer> tmp = new ArrayList<Integer>();
    public int[] reversePrint(ListNode head) {
        recur(head);
        int[] res = new int[tmp.size()];
        for(int i = 0; i < res.length; i++)
            res[i] = tmp.get(i);
        return res;
    }
    void recur(ListNode head) {
        if(head == null) return;
        recur(head.next);
        tmp.add(head.val);
    }
}
```

`辅助栈`

```java
class Solution {
    public int[] reversePrint(ListNode head) {
        LinkedList<Integer> stack = new LinkedList<Integer>();
        while(head != null) {
            stack.addLast(head.val);
            head = head.next;
        }
        int[] res = new int[stack.size()];
        for(int i = 0; i < res.length; i++)
            res[i] = stack.removeLast();
    return res;
    }
}
```

### [09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

- 栈无法实现队列功能： 栈底元素（对应队首元素）无法直接删除，需要将上方所有元素出栈。
- 双栈可实现列表倒序： 设有含三个元素的栈 A = [1,2,3]A=[1,2,3] 和空栈 B = []B=[]。若循环执行 AA 元素出栈并添加入栈 BB ，直到栈 AA 为空，则 A = []A=[] , B = [3,2,1]B=[3,2,1] ，即 栈 BB 元素实现栈 AA 元素倒序 。
- 利用栈 BB 删除队首元素： 倒序后，BB 执行出栈则相当于删除了 AA 的栈底元素，即对应队首元素。

```java
class CQueue {
    Stack<Integer> A, B;
    public CQueue() {
		A = new Stack<>();
        B = new Stack<>();
	}
    public void appendTail(int value) {
        A.push(value);
    }
    public int deleteHead() {
        if (B.isEmpty()) {
             while (!A.isEmpty()) B.push(A.pop());
        }     
        return B.isEmpty() ? -1 : B.pop();
    }
    	
}
```

### [20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

```java
class Solution {
    public boolean isNumber(String s) {
        if(s == null || s.length() == 0) return false; // s为空对象或 s长度为0(空字符串)时, 不能表示数值
        boolean isNum = false, isDot = false, ise_or_E = false; // 标记是否遇到数位、小数点、‘e’或'E'
        char[] str = s.trim().toCharArray();  // 删除字符串头尾的空格，转为字符数组，方便遍历判断每个字符
        for(int i=0; i<str.length; i++) {
            if(str[i] >= '0' && str[i] <= '9') isNum = true; // 判断当前字符是否为 0~9 的数位
            else if(str[i] == '.') { // 遇到小数点
                if(isDot || ise_or_E) return false; // 小数点之前可以没有整数，但是不能重复出现小数点、或出现‘e’、'E'
                isDot = true; // 标记已经遇到小数点
            }
            else if(str[i] == 'e' || str[i] == 'E') { // 遇到‘e’或'E'
                if(!isNum || ise_or_E) return false; // ‘e’或'E'前面必须有整数，且前面不能重复出现‘e’或'E'
                ise_or_E = true; // 标记已经遇到‘e’或'E'
                isNum = false; // 重置isNum，因为‘e’或'E'之后也必须接上整数，防止出现 123e或者123e+的非法情况
            }
            else if(str[i] == '-' ||str[i] == '+') { 
                if(i!=0 && str[i-1] != 'e' && str[i-1] != 'E') return false; // 正负号只可能出现在第一个位置，或者出现在‘e’或'E'的后面一个位置
            }
            else return false; // 其它情况均为不合法字符
        }
        return isNum;
    }
}
```

### [24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

`iteration` save curr.next as tempNext and make curr point at prev then move forward O(N) O(1)

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode currNext = curr.next;
            //当前节点指向前一个
            curr.next = prev;
            //后移
            prev = curr;
            curr = currNext;
        }
        return prev;
    }
}
```

`recursion`  O(N) O(N) space depends on the size of stack

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
}
```

### [30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

辅助栈 维护当前栈顶最小值

```java
class Solution {
    Deque<Integer> s1;
    Deque<Integer> s2;
	public minStack() {
 		s1 = new ArrrayDeque<>();
        s2 = new ArrayDeque<>();
        s2.push(Integer.MAX_VALUE);
    }
    public void push(int x) {
        s1.push(x);
        s2.push(Math.max(s2.peek(),x));
    }
    public void pop() {
        s1.pop();
        s2.pop();
    }
    public int top() {
		return s1.peek();
    }
    public int min() {
		return s2.
    }
}
```
### [35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

`hash table`

```java
class Solution {
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Node cur = head;
        Map<Node, Node> map = new HashMap<>();
        // 3. 复制各节点，并建立 “原节点 -> 新节点” 的 Map 映射
        while(cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        // 4. 构建新链表的 next 和 random 指向
        while(cur != null) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        // 5. 返回新链表的头节点
        return map.get(head);
    }
}
```

`原地置换`

思路分为第三步

+ 遍历原链表 复制原链表的每个节点到该节点之后    `1->1->2->2->3->3`
+ 遍历该链表 对每个新节点复制原节点（前驱节点）的random指向（注意两点 一是对random指向节点做判空处理 二是新节点复制的random指向是对应的random.next节点 即旧的指向旧的 新的指向新的）
+ 对该链表进行拆分（注意一点 拆分之后 原链表的最后一个节点还指向着新链表的最后一个节点 所以需要还原（也就是最终不能改动原链表）将原链表最后一个节点指向null

```java
class Solution {
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Node cur = head;
        // 1. 复制各节点，并构建拼接链表
        while(cur != null) {
            Node tmp = new Node(cur.val);
            tmp.next = cur.next;
            cur.next = tmp;
            cur = tmp.next;
        }
        // 2. 构建各新节点的 random 指向
        cur = head;
        while(cur != null) {
            if(cur.random != null)
                cur.next.random = cur.random.next;
            cur = cur.next.next;
        }
        // 3. 拆分两链表
        cur = head.next;
        Node pre = head, res = head.next;
        while(cur.next != null) {
            pre.next = pre.next.next;
            cur.next = cur.next.next;
            pre = pre.next;
            cur = cur.next;
        }
        pre.next = null; // 单独处理原链表尾节点
        return res;      // 返回新链表头节点
    }
}
```

### [58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

P.S. 如果只能用String 就用字符串拼接代替SB

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        return s.substring(n, s.length()) + s.substring(0, n);
    }
}
```

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        StringBuilder res = new StringBuilder();
        for(int i = n; i < s.length(); i++)
            res.append(s.charAt(i));
        for(int i = 0; i < n; i++)
            res.append(s.charAt(i));
        return res.toString();
    }
}
```

```java
//利用求余运算，可以简化代码
class Solution {
    public String reverseLeftWords(String s, int n) {
        StringBuilder res = new StringBuilder();
        for(int i = n; i < n + s.length(); i++)
            res.append(s.charAt(i % s.length()));
        return res.toString();
    }
}
```

### [59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

O(n) time

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length ==0) return nums;
        int n = nums.length;
        int[] res = new int[n - k + 1];
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
			//remove element out of range k
            if (!dq.isEmpty() && dq.peek() < i - k + 1) dq.poll();
            //remove smaller element in range as they are useless
            while (!dq.isEmpty() && nums[i] >= nums[dq.peekLast()]) dq.pollLast();
            dq.offer(i);
            //add result
            if (i - k + 1 >= 0) res[i - k + 1] = nums[dq.peek()];
      	}
        return res;
    }
}
```
### [59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

本题的关键性质： 当一个元素进入队列的时候，它前面所有比它小的元素就不会对答案产生影响，也就是push_back方法中while循环的逻辑 `q2.peekLast() <-> value`

注意Integer 类型比较 == 和 equals 的区别 以及常量池 方法区

```java
class MaxQueue {
    Queue<Integer> queue;
    Deque<Integer> deque;
    public MaxQueue() {
        queue = new LinkedList<>();
        deque = new LinkedList<>();
    }
    public int max_value() {
        return deque.isEmpty() ? -1 : deque.peekFirst();
    }
    public void push_back(int value) {
        queue.offer(value);
        while(!deque.isEmpty() && deque.peekLast() < value)
            deque.pollLast();
        deque.offerLast(value);
    }
    public int pop_front() {
        if(queue.isEmpty()) return -1;
        if(queue.peek().equals(deque.peekFirst())) //超过127 之后的==比较地址 返回的一定不是相同的对象
            deque.pollFirst();
        return queue.poll();
    }
}
```

### [67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

关键点 如何处理超过整数范围的数字字符串

int 范围 `-2^31 ~  2^ 31 - 1`即    -2147483648 - 2147483647

关键判断 ： res > bndry || res == bndry && c[j] > '7'

为什么是大于7 因为即使是负数 然后最后一位为8 还是会返回Integer.MIN_VALUE  即-2147483648

```java
class Solution {
    public int strToInt(String str) {
        char[] c = str.trim().toCharArray();
        if(c.length == 0) return 0;
        int res = 0, bndry = Integer.MAX_VALUE / 10;
        int i = 1, sign = 1;
        if(c[0] == '-') sign = -1;
        else if(c[0] != '+') i = 0;
        for(int j = i; j < c.length; j++) {
            if(c[j] < '0' || c[j] > '9') break;
            if(res > bndry || res == bndry && c[j] > '7') return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            res = res * 10 + (c[j] - '0');
        }
        return sign * res;
    }
}
```

## :banana:动态规划 10
### [10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

```java
class Solution {
    public int fib(int n) {
        int p = 0, q = 0, r = 1;
        for (int i = 0; i < n; i++) {
            p = q;
            q = r;
            r = (p + q) % 1000000007;
        }
        return q;
    }
}
```
### [10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

`Fibonacci`

注意缓存 拒绝傻递归

```java
class Solution {
    public int numWays(int n) {
		int p = 0, q = 0, r = 1;
        for (int i = 0; i < n; i++) {
            p = q;
            q = r;
            r = (p + q) % 1000000007;
        }
        return r;
    }
}
```
### [19. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

![image-20211202225426016](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20211202225426016.png)

正则表达式p的情况

+ 如果为  '*'
  + 如果s当前字符和 p的前一个字符不匹配  `dp[i][j] = dp[i][j - 2] ` 
    + 表示 忽略p的前一位 即前一位字符的出现次数为0    ---》   ab*
  + 如果s当前字符和 p的前一个字符匹配 `dp[i][j] = dp[i - 1][j]`
    + 如果当前s的 字符 == 当前p的前一个字符  即前一位字符的出现次数为1  ----》a*
    + 如果当前p的字符为 ' * ' 且 p的上一个字符为 ' . '    ---》 a.*  即当前这一位无论如何都是可以匹配的 . * 
+ 否则
  + `dp[i][j] = dp[i - 1][j - 1]`
    + 表示当前p和q的字符对应 a / a
    + 当前p的字符为 ' . ' 所以当前位一定能对应上 只考虑s之前和p之前

```java
class Solution {
    public boolean isMatch(String s, String p) {
        int m = s.length(),n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    //忽略掉p的后边两位
                    dp[i][j] = dp[i][j - 2];
                  //如果p的前一位和s当前位 对应有可能是字符匹配也可能是'.'匹配
                    if (matches(s, p, i, j - 1)) {
                        //同时忽略掉s的当前位
                        dp[i][j] = dp[i][j] || dp[i - 1][j];
                    }
                } else {
                    //如果s的当前位和p的当前位匹配
                    if (matches(s, p, i, j)) {
                        //忽略掉s和p当前位
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                }
            }
        }
        return dp[m][n];
    }
    boolean matches(String s, String p, int i, int j) {
        if (i == 0) return false;
        else if (p.charAt(j - 1) == '.') return true;
        else return s.charAt(i - 1) == p.charAt(j - 1);
    }
}
```

### [42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

`dp` O(N) time O(1) space

状态定义： dp[i] 代表以元素nums[i] 为结尾的连续子数组最大和

状态转移方程：

    case dp[i - 1] > 0 		dp = dp[i - 1] + nums[i]
    case dp[i - 1] <= 0		 dp = nums[i]

```java
class Solution {
	public int maxSubArray(int[] nums) {
		int res = nums[0];
        for (int i = 1; i < nums.length ;i++) {
            nums[i] += Math.max(nums[i - 1], 0);
            res = Math.max(res, nums[i]);
        }
        return res;
    }
}
```

`optimize dp`

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int res = Integer.MIN_VALUE, sum = 0;
        for (int num : nums) {
            if (sum > 0) sum += num;
            else sum =num;
            res = Math.max(res, sum);
        }
        return res;
    }
}
```

### [46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

`dp`

```java
class Solution {
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= s.length(); i ++){
            String temp = s.substring(i-2, i);
            if(temp.compareTo("10") >= 0 && temp.compareTo("25") <= 0)
                dp[i] = dp[i-1] + dp[i-2];
            else
                dp[i] = dp[i-1];
        }
        return dp[s.length()];
    }
}
```

>其中 dp[i] 只与 前两项有关 所以可以使用变量代替 然后向后迭代

`dp` 通过取余的方式从右往左进行判断节省了空间

```java
class Solution {
    public int translateNum(int num) {
        int a = 1, b = 1, x, y = num % 10;
        while(num != 0) {
            num /= 10;
            x = num % 10;
            int tmp = 10 * x + y;
            int c = (tmp >= 10 && tmp <= 25) ? a + b : a;
            b = a;
            a = c;
            y = x;
        }
        return a;
    }
}

```

### [47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

`dp`

定义一个二维数组 表示走到当前dp[ i ] [ j ]时 最大的礼物数

`状态转移方程`

```java
1. i == 0 && j == 0  dp[i][j] = grid[i][j];
2. i == 0 && j != 0  dp[i][j] = dp[i][j - 1] + grid[i][j];
3. j == 0 && i != 0  dp[i][j] = dp[i - 1][j] + grid[i][j];
4. dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
```

```java
class Soluiton {
    public int maxValue(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = i == 0 && j == 0 ? grid[0][0] : (i == 0 ? dp[i][j - 1] : dp[i - 1][j]) + grid[i][j];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
                }
            }
        }
        return dp[m - 1][n - 1];
    }
}
```

**对于处理边界问题 可以扩大一层边界 这样可以简化dp方程**

```java
class Solution {
    public int maxValue(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1];
            }
        }
        return dp[1][1];
    }
}
```

**由于dp方程只与上一层dp有关所以可以优化空间复杂度 直接在grid数组上进行dp操作**

### [48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

`dp`

**状态定义**

设动态规划列表 dp ，dp*[*j*] 代表以字符 s[j]*s*[*j*] 为结尾的 “最长不重复子字符串” 的长度。

**状态转移方程**

固定右边界 j*j* ，设字符 s[j]*s*[*j*] 左边距离最近的相同字符为 s[i]*s*[*i*] ，即 s[i] = s[j]*s*[*i*]=*s*[*j*]

- dp[j -1] < j - i 
  - dp[j - 1] + 1
- dp[j - 1] >= j - 1
  - j - i

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> dic = new HashMap<>();
        int res = 0, temp = 0;
        for (int j = 0; j < s.length(); j++) {
            int i = dic.getOrDefault(s.charAt(j), -1); //get index i
            dic.put(s.charAt(j), j); //update hashSet
            temp = temp < j - i ? temp + 1 : j - i; //dp[j - 1] -> dp[j]
            res = Math.max(res, temp); //max(dp[j - 1], dp[j])
        }
        return res;
    }
}
```

`滑动窗口`

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int[] dic = new int[128];
        //res 记录当当前子字符串的最大长度
        //idx 记录当前下标对应字符的最大索引值 如果没有出现则为0
        int res = 0, idx = 0;	
        for (int i = 0; i < s.length(); i++) {
            //更新当前字符的最大索引值
            idx = Math.max(idx, dic[s.charAt(i)]);
            //更新索引表
            dic[s.charAt(i)] = i + 1; //窗口要后移一位
            //更新当前最大值
            res = Math.max(res, i - idx + 1); //下标表示区间长度要加一
        }
        return res;
    }
}
```

### [ 49. 丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

`dp` **丑数** 就是只包含质因数 `2`、`3` 和/或 `5` 的正整数。

dp[i] 为第i个丑数 保存三个变量作为下标存储2 3 5 的位置 每一轮取该下标对应的最小的丑数，更新到dp[i]中 对应更新的丑数因子下标++

```java
class Solution {
    public int nthUglyNumber(int n) {
        int[] dp = new int[n];
        int a = 0, b = 0, c = 0;
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            int n1 = dp[a] * 2, n2 = dp[b] * 3, n3 = dp[c] * 5;
            dp[i] = n1 < n2 ? (n1 < n3 ? n1 : n3) : (n2 < n3 ? n2 : n3);
            //注意这里的if判断 没有else 因为如果当前丑数为2 3 5 的公倍数，则对应的2 3 5的下标都要增加 而不是只增加一个 否则下一个丑数就还是由没加的下标产生的和上一个相同的丑数😄
            if (dp[i] == n1) a++;
            if (dp[i] == n2) b++;
            if (dp[i] == n3) c++; 
        }
        return dp[n - 1];
    }
}
```



### [60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

`dp` 

逆向推 骰子数为n的结果 = （n - 1）的 六个结果的和 但是某些情况不是六个结果 （会出现越界） X 

正向推 已知 f(n - 1) 如何求f(n)  √

	遍历 f(n - 1) 的结果 分别对 新的f(n) 进行累加 累加后 形成新的dp数组 然后指向它

细节 

+ 当骰子数为x时 下一层骰子数 为 x + 1 骰子点数 为 5 * x + 1 (也就是下一层dp数组的长度)
+ 下一层要循环 x 次 (即上一层dp数组的长度)
+ 每一次循环要累计6种结果 (即骰子点数的最大似然概率)

```java
class Solution {
    public double[] dicesProbability(int n) {
        double[] dp = new double[6];
        Arrays.fill(dp, 1. / 6.0);
        for (int i = 2; i <= n; i++) {
            double[] temp = new double[5 * i + 1];
            for (int j = 0; j < dp.length; j++) {
                for (int k = 0; k < 6; k++) {
                    temp[j + k] += dp[j] / 6;
                }
            }
            dp = temp;
        }
        return dp;
    }
}
```



### [63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

`dp`

```java
class Solution {
    public int maxProfit(int[] prices) {
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        for (int i = 0; i < prices.length; i++) {
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, -prices[i]);
        }
        return dp_i_0;
    }
}
```



## :orange:搜索与回溯算法 18

### [12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

`DFS`

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        char[] words = word.toCharArray();
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                if(dfs(board, words, i, j, 0)) return true;
            }
        }
        return false;
    }
    boolean dfs(char[][] board, char[] word, int i, int j, int k) {
        if(i >= board.length || i < 0 || j >= board[0].length || j < 0 || board[i][j] != word[k]) return false;
        if(k == word.length - 1) return true;
        board[i][j] = '\0';
        boolean res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) || 
                      dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i , j - 1, k + 1);
        board[i][j] = word[k];
        return res;
    }
}
```



### [13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

1.求数位和

```java
int sums(int x) {
    int s = 0;
    while (x != 0) {
        s += x % 10;
        x /= 10;
    }
    return s;
}
```

2.数位和增量公式 仅适用于 1 <=n, m <= 100

	设x的数位和为s1，x+1的数位和为s2

- 当(x + 1) % 10 = 0 时 s2 = s1 - 8
- 当(x + 1) % 10 ！= 0 时，s2 = s1 + 1

```java
s_x2 = (x + 1) % 10 != 0 ? s_x + 1 : s_x - 8;
```

`DFS`

```java
class Solution {
    int m, n, k;
    boolean[][] visited;
    public int movingCount(int m, int n, int k) {
        this.m = m; this.n = n; this.k = k;
        this.visited = new boolean[m][n];
        return dfs(0, 0, 0, 0);
    }
    public int dfs(int i, int j, int si, int sj) {
        if(i >= m || j >= n || k < si + sj || visited[i][j]) return 0;
        visited[i][j] = true;
        return 1 + dfs(i + 1, j, (i + 1) % 10 != 0 ? si + 1 : si - 8, sj) + dfs(i, j + 1, si, (j + 1) % 10 != 0 ? sj + 1 : sj - 8);
    }
}
```

### [26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

注意判断顺序 java的短路特性 先判断null 否则报空指针异常

```java
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        return (A != null && B != null) && (recur(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B));
    }
    boolean recur(TreeNode A, TreeNode B) {
        if(B == null) return true;
        if(A == null || A.val != B.val) return false;
        return recur(A.left, B.left) && recur(A.right, B.right);
    }
}
```

### [27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

`recursion`

```java
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if(root == null) return null;
        TreeNode tmp = root.left;
        root.left = mirrorTree(root.right);
        root.right = mirrorTree(tmp);
        return root;
    }
}
```

`iteration`

```java
public TreeNode mirrorTree(TreeNode root) {
        if (root == null) return null;
        Deque<TreeNode> queue = new LinkedList<>(){{
            add(root);
        }};
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.right != null) queue.add(node.right);
            if (node.left != null) queue.add(node.left);
            TreeNode temp = node.left;
            node.left = node.right;
            node.right = temp;
        }
        return root;
    }
```

### [28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

`recursion`

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return recur(root.left, root.right);
    }
    boolean recur(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        else if (left == null || right == null) return false;
        else if (left.val != right.val) return false;
        return recur(left.left, right.right) && recur(left.right, right.left);
    }
}
```

`iteration`

```java
class Solution {
    int INF = 0x3f3f3f3f;
    TreeNode emptyNode = new TreeNode(INF);
    boolean isSymmetric(TreeNode root) {
        if (root == null) return true;

        Deque<TreeNode> d = new ArrayDeque<>();
        d.add(root);
        while (!d.isEmpty()) {
            // 每次循环都将下一层拓展完并存到「队列」中
            // 同时将该层节点值依次存入到「临时列表」中
            int size  = d.size();
            List<Integer> list = new ArrayList<>();
            while (size-- > 0) {
                TreeNode poll = d.pollFirst();
                if (!poll.equals(emptyNode)) {
                    d.addLast(poll.left != null ? poll.left : emptyNode);
                    d.addLast(poll.right != null ? poll.right : emptyNode);
                }
                list.add(poll.val);
            }
            
            // 每一层拓展完后，检查一下存放当前层的该层是否符合「对称」要求
            if (!check(list)) return false;
        }
        return true;
    }

    // 使用「双指针」检查某层是否符合「对称」要求
    boolean check(List<Integer> list) {
        int l = 0, r = list.size() - 1;
        while (l < r) {
            if (!list.get(l).equals(list.get(r))) return false;
            l++;
            r--;
        }
        return true;
    }
}
```



### [32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

`BFS` 借助队列实现

```java
class Solution {
    public int[] levelOrder(TreeNode root) {
        if(root == null) return new int[0];
        Queue<TreeNode> queue = new LinkedList<>(){{ add(root); }};
        ArrayList<Integer> ans = new ArrayList<>();
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            ans.add(node.val);
            if(node.left != null) queue.add(node.left);
            if(node.right != null) queue.add(node.right);
        }
        int[] res = new int[ans.size()];
        for(int i = 0; i < ans.size(); i++)
            res[i] = ans.get(i);
        return res;
    }
}
```

### [32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

I. 按层打印： 题目要求的二叉树的 从上至下 打印（即按层打印），又称为二叉树的 广度优先搜索（BFS）。BFS 通常借助 队列 的先入先出特性来实现。

II. 每层打印到一行： 将本层全部节点打印到一行，并将下一层全部节点加入队列，以此类推，即可分为多行打印。

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Deque<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> list = new ArrayList<>();
        if (root != null) queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            for (int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                temp.add(node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            list.add(temp);
        }
        return list;
    }
}
```

### [32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

`方法一：层序遍历 + 双端队列`

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Deque<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> list = new ArrayList<>();
        if (root != null) queue.add(root);
        while (!queue.isEmpty()) {
            LinkedList<Integer> temp = new LinkedList<>();
            for (int i = queue.size(); i > 0; i--){
                TreeNode node = queue.poll();
                if (list.size() % 2 == 0) temp.addLast(node.val);
                else temp.addFirst(node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            list.add(temp);
        }          
        return list;
    }
}
```

`方法二：层序遍历 + 双端队列（奇偶层逻辑分离）` 奇偶层逻辑分离 减少了N次冗余判断

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Deque<TreeNode> deque = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if(root != null) deque.add(root);
        while(!deque.isEmpty()) {
            // 打印奇数层
            List<Integer> tmp = new ArrayList<>();
            for(int i = deque.size(); i > 0; i--) {
                // 从左向右打印
                TreeNode node = deque.removeFirst();
                tmp.add(node.val);
                // 先左后右加入下层节点
                if(node.left != null) deque.addLast(node.left);
                if(node.right != null) deque.addLast(node.right);
            }
            res.add(tmp);
            if(deque.isEmpty()) break; // 若为空则提前跳出
            // 打印偶数层
            tmp = new ArrayList<>();
            for(int i = deque.size(); i > 0; i--) {
                // 从右向左打印
                TreeNode node = deque.removeLast();
                tmp.add(node.val);
                // 先右后左加入下层节点
                if(node.right != null) deque.addFirst(node.right);
                if(node.left != null) deque.addFirst(node.left);
            }
            res.add(tmp);
        }
        return res;
    }
}
```

`方法三：层序遍历 + 倒序`

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if(root != null) queue.add(root);
        while(!queue.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();
            for(int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                tmp.add(node.val);
                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
            if(res.size() % 2 == 1) Collections.reverse(tmp);
            res.add(tmp);
        }
        return res;
    }
}

```

### [34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

`回溯`

```java
class Solution {
    //多态
    LinkedList<Integer> temp = new LinkedList<>();
    List<List<Integer>> list = new LinkedList<>();
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        recur(root, target);
        return list;	
    }
    void recur(TreeNode node, int target) {
        //recursion terminator
        if (node == null) return;
        //process current logic
        temp.add(node.val);
        target -= node.val;
        if (target == 0 && node.left == null && node.right == null) list.add(new LinkedList<Integer>(temp));
        recur(node.left, target);
        recur(node.right, target);
        //回溯 清除当前操作 
        temp.removeLast();
    }
}
```
### [36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

二叉树的中序遍历

```java
class Solution {
    Node head, pre;
    public Node treeToDoublyList(Node root) {
        if(root==null) return null;
        dfs(root);
        pre.right = head;
        head.left =pre;//进行头节点和尾节点的相互指向，这两句的顺序也是可以颠倒的
        return head;
    }

    public void dfs(Node cur){
        if(cur==null) return;
        dfs(cur.left);
        //pre用于记录双向链表中位于cur左侧的节点，即上一次迭代中的cur,当pre==null时，cur左侧没有节点,即此时cur为双向链表中的头节点
        if(pre==null) head = cur;
        //反之，pre!=null时，cur左侧存在节点pre，需要进行pre.right=cur的操作。
        else pre.right = cur;      
        cur.left = pre;//pre是否为null对这句没有影响,且这句放在上面两句if else之前也是可以的。
        pre = cur;//pre指向当前的cur
        dfs(cur.right);//全部迭代完成后，pre指向双向链表中的尾节点
    }
}
```

### [37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

`层序序列化`

同时需要注意 要保留每个节点的左右子节点信息 从而保证唯一性

注意点： 以root节点为索引 0 

索引n 的节点的 left 节点 索引为 2 * (n - m) + 1, right节点 索引为 2 * (n - m) + 2; 其中m为0 - n 索引中 节点为null-的个数

```java
public class Codec {
    public String serialize(TreeNode root) {
        if(root == null) return "[]";
        StringBuilder res = new StringBuilder("[");
        Queue<TreeNode> queue = new LinkedList<>() {{ add(root); }};
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if(node != null) {
                res.append(node.val + ",");
                queue.add(node.left);
                queue.add(node.right);
            }
            else res.append("null,");
        }
        res.deleteCharAt(res.length() - 1);
        res.append("]");
        return res.toString();
    }

    public TreeNode deserialize(String data) {
        if(data.equals("[]")) return null;
        String[] vals = data.substring(1, data.length() - 1).split(",");
        TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
        Queue<TreeNode> queue = new LinkedList<>() {{ add(root); }};
        int i = 1;
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if(!vals[i].equals("null")) {
                node.left = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.left);
            }
            i++;
            if(!vals[i].equals("null")) {
                node.right = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.right);
            }
            i++;
        }
        return root;
    }
}

```

### [38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

`dfs + 剪枝`

```java
class Solution {
	List<String> res = new LinkedList<>();
    char[] c;
    public String[] permutation(String s) {
        c = s.toCharArray();
        dfs(0);
        return res.toArray(new String[res.size()]);
    }
    
    void dfs(int x) {
        //recursion terminator
        if (x == c.length - 1){
            //process result
            res.add(String.valueOf(c));
            return;
        }
        //process current logic
        HashSet<Character> set = new HashSet<>();
        for (int i = x; i < c.length; i++) {
			if (set.contains(c[i])) continue;
            set.add(c[i]);
            swap(i,x);
            //drill down
            dfs(x + 1);
            //restore current status
            swap(i,x);
        }
    }
    
    void swap(int a, int b) {
        char tmp = c[a];
        c[a] = c[b];
        c[b] = tmp;
    }
}
```

### [54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

二叉搜索树的逆序中序遍历第k个节点

注意点  维护成员变量k  因为递归 k状态不一致

```java
class Solution {
    int res, k;
    public int kthLargest(TreeNode root, int k) {
        this.k = k;
        recur(root); 
        return res;
    }
    void recur (TreeNode root) {
        if (root == null || k == 0) return;
        recur(root.right);
        if (--k == 0) res = root.val;
        recur(root.left);
    }
}
```

### [55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

`recursion DFS` 

> 即求左右子树的较大值

```java
class Solution {
 	public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
	}   
}
```

`queue BFS`

>每遍历一层，则计数器 +1 ，直到遍历完成，则可得到树的深度。

```java
class Solution {
   public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int dep = 0;
        Deque<TreeNode> queue = new LinkedList<>(), temp;
        queue.add(root);
        while (!qu eue.isEmpty()) {
            temp = new LinkedList<>();
            for (TreeNode node : queue) {
                if(node.left != null) temp.add(node.left);
                if (node.right != null) temp.add(node.right);    
            }
            queue = temp;
            dep += 1;
        }
        return dep;
    }
}
```

### [55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

`自底向上` 比自顶向下避免了重复运算

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return recur(root) != -1;
    }

    private int recur(TreeNode root) {
        if (root == null) return 0;
        int left = recur(root.left);
        if(left == -1) return -1;
        int right = recur(root.right);
        if(right == -1) return -1;
        return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
    }
}
```

### [64. 求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

`math`

```java
class Solution {
    public int sumNums(int n) {
        return (1 + n) * n / 2;
    }
}
```

### [68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

`recursion` O(N) time O(1) space

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //如果小于等于0，说明p和q位于root的两侧，直接返回即可
        if ((root.val - p.val) * (root.val - q.val) <= 0) return root;
        //否则，p和q位于root的同一侧，就继续往下找
        return lowestCommonAncestor(p.val < root.val ? root.left : root.right, p, q);
    }
}
```

### [68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

`recursion` O(N) time O(N) space 

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        return root;
    }
}
```

## :pear:分治算法 5

### [07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

`recursion--divide & conquer`

O(N) time / O(N) space (best O(logN) space---full b tree)

根的确定是根据 先序遍历的pre_root确定的 当确定了根节点 就可以根据中序遍历确定左子树和右子树的左右边界

```java
class Solution {
    HashMap<Integer, Integer> map = new HashMap<>();//标记中序遍历
    int[] preorder;//保留的先序遍历，方便递归时依据索引查看先序遍历的值

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        //将中序遍历的值及索引放在map中，方便递归时获取左子树与右子树的数量及其根的索引
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        //三个索引分别为
        //当前根的的索引
        //递归树的左边界，即数组左边界
        //递归树的右边界，即数组右边界
        return recur(0,0,inorder.length-1);
    }

    TreeNode recur(int pre_root, int in_left, int in_right){
        if(in_left > in_right) return null;// 相等的话就是自己
        TreeNode root = new TreeNode(preorder[pre_root]);//获取root节点
        int idx = map.get(preorder[pre_root]);//获取在中序遍历中根节点所在索引，以方便获取左子树的数量
        //左子树的根的索引为先序中的根节点+1 
        //递归左子树的左边界为原来的中序in_left
        //递归右子树的右边界为中序中的根节点索引-1
        root.left = recur(pre_root+1, in_left, idx-1);
        //右子树的根的索引为先序中的 当前根位置 + 左子树的数量 + 1
        //递归右子树的左边界为中序中当前根节点+1
        //递归右子树的有边界为中序中原来右子树的边界
        root.right = recur(pre_root + (idx - in_left) + 1, idx+1, in_right);
        return root;

    }
}
```

### [16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

`分治` 求 x 的 n次幂 == 求 x * x 的 (n >> 1) 次幂

Java 代码中 int32 变量n∈[−2147483648,2147483647] ，因此当 n = -2147483648 时执行 n=−n 会因越界而赋值出错。两种解决方案：第一种是n用long来存第二种是提前判断是否是Integer.MIN_VALUE 如果是 则先除2

处理n<0的问题 当n小于0时，n = -n (注意上面问题，Integer.MIN_VALUE赋值会越界), x = 1 / x

```java
class Solution {
    public double myPow(double x, int n) {
        if (n == 0) return 1.00;
        if (n == Integer.MIN_VALUE) {
            x *= x;
            n >>= 1;
        }
        if (n < 0) {
            n = -n;
            x = 1 / x;
        }
        return (n & 1) == 0 ? myPow(x * x, n >> 1) : x * myPow(x * x, n >> 1);
    }
}
```

### [17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

n 和 最后一个数字的关系为 n == 10 ^ n - 1

如果int的范围没有超过最大值 

```java
class Solution {
    public int[] printNumbers(int n) {
        int size = (int)Math.pow(10, n) - 1;
        int[] res = new int[size];
        for (int i = 0; i < size; i++) {
            res[i] = i + 1;
        }
        return res;
    }
}
```

考虑`大数`的情况 （面试实际场景）

```java
class Solution {
    char[] num;
    int[] ans;
    int count = 0,n;
    public int[] printNumbers(int n) {
        this.n = n;
        num = new char[n];
        ans = new int[(int) (Math.pow(10, n) - 1)];
        dfs(0);
        return ans;
    }
    private void dfs(int n) {
        if (n == this.n) {
            String tmp = String.valueOf(num);
            int curNum = Integer.parseInt(tmp);
            if (curNum!=0) ans[count++] = curNum;
            return;
        }
        for (char i = '0'; i <= '9'; i++) {
            num[n] = i;
            dfs(n + 1);
        }
    }
}
```

### [33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        // 单调栈使用，单调递增的单调栈
        Deque<Integer> stack = new LinkedList<>();
        // 表示上一个根节点的元素，这里可以把postorder的最后一个元素root看成无穷大节点的左孩子
        int pervElem = Integer.MAX_VALUE;
        // 逆向遍历，就是翻转的先序遍历
        for (int i = postorder.length - 1;i>=0;i--){
            // 左子树元素必须要小于递增栈被peek访问的元素，否则就不是二叉搜索树
            if (postorder[i] > pervElem){
                return false;
            }
            while (!stack.isEmpty() && postorder[i] < stack.peek()){
                // 数组元素小于单调栈的元素了，表示往左子树走了，记录下上个根节点
                // 找到这个左子树对应的根节点，之前右子树全部弹出，不再记录，因为不可能在往根节点的右子树走了
                pervElem = stack.pop();
            }
            // 这个新元素入栈
            stack.push(postorder[i]);
        }
        return true;
    }
}
```

### [51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

逆序对：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。 mergeSort 的思想

`这个题的本质就是求归并排序交换（升序）的次数` mergeSort

`merge` O(nlog(n))

```java
class Solution {
    public int reversePairs(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        return mergeSort(nums, 0, nums.length - 1);
    }
    int cnt = 0;
    private int mergeSort(int[] nums, int left, int right) {
        if (right <= left) return 0;
        int mid = left + right >> 1;
        cnt = mergeSort(nums, left, mid) + mergeSort(nums, mid + 1, right);
        merge(nums, left, mid, right);
        return cnt;
    }
    private void merge(int[] nums, int left, int mid , int right) {
        int i = left, j = mid + 1, k = 0;
        int[] temp = new int[right - left + 1];
        while (i <= mid) {
            while (j <= right && nums[j] < nums[i]) temp[k++] = nums[j++];
            cnt+= j - (mid + 1);
            temp[k++] = nums[i++];
        }
        while (j <= right) temp[k++] = nums[j++];
        System.arraycopy(temp, 0, nums, left, temp.length);
    }
}
```

## :watermelon:排序 4

### [40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

`quick select`

思想类似于快排，但是分区之后不排序

- k == m 直接返回左数组
- k < m 返回长度为k的左子数组
- k > m 返回左数组 + (k - m)继续快速选择右数组

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (arr.length <= k) return Arrays.copyOf(arr, k);
        int begin = 0, end = arr.length - 1;
        return quickSort(arr, begin, end, k);
    }
    private int[] quickSort(int[] arr, int begin, int end, int k) {
        int pivot = partition(arr, begin, end);
        if (pivot == k) return Arrays.copyOf(arr, k);
        return pivot < k ? 
            quickSort(arr, pivot + 1, end, k) : quickSort(arr, begin, pivot - 1,k);
    }
    private int partition(int[] arr, int begin, int end) {
        int pivot = end, counter = begin;
        for (int i = begin; i < end; i++) {
            if (arr[i] < arr[pivot]) {
                swap(arr,i, counter);
                counter++;
            }
        }
        swap(arr, pivot, counter);
        return counter;
    }
    void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```

### [41. 数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

`priorityQueue`

A: 小顶堆  保证首部是最小的且大于B的首部

B：大顶堆 保证首部最大

**如何保证A的所有元素比B大呢**

A先addNum 然后再弹出队首 此时保证队首是最小的 将其添加到B 当A.size() == B.size()时同理

```java
class MedianFinder {
    PriorityQueue<Integer> A;
    PriorityQueue<Integer> B;
    /** initialize your data structure here. */
    public MedianFinder() {
        A = new PriorityQueue<>();
        B = new PriorityQueue<>((x, y) -> (y - x));
    }
    
    public void addNum(int num) {
        if (A.size() != B.size()) {
            A.add(num);
            B.add(A.poll());
        } else {
            B.add(num);
            A.add(B.poll());
        }
    }
    
    public double findMedian() {
        return A.size() == B.size() ? (A.peek() + B.peek()) * 1.0 / 2 : A.peek();
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```

### [45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

此题求拼接起来的最小数字，本质上是一个排序问题。设数组 numsnums 中任意两数字的字符串为 xx 和 yy ，则规定 排序判断规则 为：

若拼接字符串 x + y > y + xx+y>y+x ，则 xx “大于” yy ；
反之，若 x + y < y + xx+y<y+x ，则 xx “小于” yy ；

x* “小于” y*y* 代表：排序完成后，数组中 x*x* 应在 y*y* 左边；“大于” 则反之。

个人感觉 这个题的关键正确理解自定义排序的逻辑 为什么可以保证全局符合该排序逻辑

就针对快排的思想

正常快排就是给定一个pivot 保证pivot左边的元素“小于”pivot位置的元素 右边同理

而自定义排序逻辑之后 是

+ 保证左边元素+ pivot 对应的数字组成的数字 一定是小于pivotNum + 左边的数字
+ 保证右边元素 + pivot对应的数字所组成的数字 大于pivotNum + 右边的数字

按照这个逻辑进行分治递归下去 就会保证最终左右元素之间的关系一定是最优的（最小）

而这些元素的位置也就确定了下来

**O(N log N)** **time**  快排average   **O(N) space** String数组

```java
class Solution {
    public String minNumber(int[] nums) {
        String[] str = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            str[i] = String.valueOf(nums[i]);
        }
        quickSort(0, nums.length - 1, str); //Arrays.sort(arr, (x, y) -> (x + y).compareTo(y + x));
        StringBuilder sb = new StringBuilder();
        for (String s : str) sb.append(s);
        return sb.toString();
    }

    //根据自定义的排序规则进行快速排序
    void quickSort(int begin, int end, String[] arr) {
        if (end <= begin) return;
        int pivot = partition(begin, end, arr);
        quickSort(begin, pivot - 1, arr);
        quickSort(pivot + 1, end, arr);
    }
    
    private int partition(int begin, int end, String[] arr) {
        int counter = begin, pivot = end;
        for (int i = begin; i < end; i++) {
            //区别就在于此处 修改了排序的规则
            if ((arr[i] + arr[pivot]).compareTo(arr[pivot] + arr[i]) <= 0) {
                swap(counter, i, arr);
                counter++;
            }
        }
        swap(pivot, counter, arr);
        return counter;
    }
    void swap(int i, int j, String[] arr) {
        String temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```

### [61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

思路：排序后统计大小王的数量从而确定最小值的位置，过程中如果有重复，则提前返回false。最后比较max - min < 5

```java
class Solution {
    public boolean isSraight(int[] nums) {
        int joker = 0;
        Arrays.sort(nums);
        for (int i = 0 ; i < 4; i++) {
            if (nums[i] == 0) joker++;
            else if (nums[i] == nums[i + 1]) return flase;
        }
        return nums[4] - nums[joker] < 5;
    }
}
```

## :strawberry:查找算法 6

### [03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

`原地置换`如果没有重复数字，那么正常排序后，数字i应该在下标为i的位置，所以思路是重头扫描数组，遇到下标为i的数字如果不是i的话，（假设为m),那么我们就拿与下标m的数字交换。在交换过程中，如果有重复的数字发生，那么终止返回ture

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
		int temp;
         for (int i = 0; i < nums.length; i++) {
             while (nums[i] != i) {
                 if (nums[i] == nums[nums[i]]) return nums[i];
             //swap nums[i] & nums[nums[i]]
             temp = nums[i]; nums[i] = nums[temp]; nums[temp] = temp;
			}
        }
        return -1;
    }
} 
```

### [04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

从左下角开始查找 具备以下性值 右边元素均大于matrix[m] [n] 上边元素均小于matrix[m] [n]

iteration terminator :  m >=0 && n < matrix[0].length

```java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) return false;
        int m = matrix.length - 1, n = 0;
        while (m >= 0 && n < matrix[0].length) {
            if (target == matrix[m][n]) return true;
            else if (target > matrix[m][n]) n++;
            else m--;
        }
        return false;
    }
}
```

### [11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

`二分法`

> 为什么用中值判断numbers[r] 而不去判断numbers[l]

因为二分法的本质是通过判断大小来缩小区间 索引`r`一定是右侧数组的右边界 （当然有可能整个数组都是右侧数组 即旋转点x = 0）

而`l`并不能确定是在哪个排序数组中 从而不能缩小范围

> numbers[mid] == numbers[r] 的处理条件 0 1 2 2 2 
>
> 4 5 2 2 2 

另外值得一提的是 也可以通过线性遍历实际上，当出现 nums[m] = nums[j]nums[m]=nums[j] 时，一定有区间 [i, m][i,m] 内所有元素相等 或 区间 [m, j][m,j] 内所有元素相等（或两者皆满足）。对于寻找此类数组的最小值问题，可直接放弃二分查找，而使用线性查找替代。



```java
class Solution {
    public int minArray(int[] numbers) {
        int l = 0, r = numbers.length -1;
        while (l < r) {
            int mid = l + (r - l >> 1);
            if (numbers[mid] > numbers[r]) l = mid + 1;
            else if (numbers[mid] < numbers[r]) r = mid;
            else r--;
        }
        return numbers[l];
    }
}
```

```java
class Solution {
    public int minArray(int[] numbers) {
        int l = 0, r = numbers.length - 1;
        while (l < r) {
            int m = l + r >> 1;
            if (numbers[m] > numbers[r]) l = m + 1;
            else if (numbers[m] < numbers[r]) r = m;
            else {
                int p = l;
                for (int k = p + 1; k < r; k++) {
                    if(numbers[k] < numbers[p]) p = k;
                }
                return numbers[p];
            }
        }
        return numbers[l];
    }
}
```

### [ 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

`hash`

```java
class Solution {
    public char firstUniqChar(String s) {
        int[] dic = new int[26];
        for (char c : s.toCharArray()) dic[c - 'a'] += 1;
        for (char c : s.toCharArray()) {
            if (dic[c - 'a'] == 1) return c;
        }
        return ' ';
    }
}
```

### [53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

```java
class Solution {
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) return 0;
        int l = 0,r = nums.length - 1;
        while (l < r) {
            int m = l + r >> 1;
            if (nums[m] >= target) r = m;
            else l = m + 1;
        }
        if (nums[l] != target) return 0;
        int temp = l;
        else {
            l = 0;r = nums.length - 1;
            while (l < r) {
                int m = l + r + 1 >> 1;
                if (nums[m] <= target) l = m;
                else r = m - 1;
            }
            return l - temp + 1;
        }
    }
}
```
### [53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

排序数组中的搜索问题，首先要想到**二分法**解决

```java
class Solution {
    public int missingNumber(int[] nums) {
        int i = 0, j = nums.length - 1;
        while (i <= j) {
            int m = i + (j - i >> 1);
            if (nums[m] == m) i = m + 1;
            else j = m - 1;
        }
        return i;
    }
}
```

## :peach:双指针 6

### [18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

```java
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        if (head.val == val) return head.next;
        ListNode dummy = head, prev = null;
        while (head != null) {
            if (head.val == val) {
                prev.next = head.next;
                head.next = null;
            }
            prev = head;
            head = head.next;
        }
        return dummy;
    }
}
```

### [21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

`double pointer`

左指针指向奇数 右指针指向偶数 如果符合则移动到下一个 否则 交换左右指针下标对应的元素

注意事项

取2的余数可以通过位运算&1处理， 注意运算优先级

swap 可以使用异或运算

```java
void swap(int[] nums, int i, int j) {
    if (i != j) {
        nums[i] = nums[i]^nunms[j];
        nums[j] = nums[i]^nums[j];
        nums[i] = nums[i]^nums[j];
    }
}
```

```java
class Solution {
    public int[] exchange(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            //左边是奇数
            while (l < r && (nums[l] & 1) == 1) {
                l++;
            }
            while (l < r && (nums[r] & 1 )== 0) {
                r--;
            }
            swap (nums, l, r);
        }
        return nums;
    }
    private void swap(int[] nums, int i, int j) {
            if (i != j) {
                nums[i] = nums[i] ^ nums[j];
                nums[j] = nums[i] ^ nums[j];
                nums[i] = nums[i] ^ nums[j];
            }
        }
}
```
### [22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

`fast and slow pointer` 类比于移除链表中第n个节点题目 leetcode 19 

```java
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
	    int t = 0;
        ListNode slow = head, fast = head;
        while (fast != null) {
            if (t++ >= k) slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
}
```
### [25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

```java
public Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        else if (l2 == null) return l1;
        else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}
```

`iteration`

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode p, dummpy = new ListNode(0);
        p = dummpy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                p.next = l1;
                l1 = l1.next;
            } else {
                p.next = l2;
                l2 = l2.next;
            }
            p = p.next;
        }
        p.next = (l1 == null) ? l2 : l1;
        return dummpy.next;
    }
}
```


### [52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

`double pointer`

**当节点为null时 返回另一个节点的头节点** O(a + b) time O (1) space

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        //不使用两个链表的头结点 便于后续定位其引用
        ListNode A = headA, B = headB;
        while (A != B) {
            A = A != null ? A.next : headB;
            B = B != null ? B.next : headA;
        }
        return A;
    }
}
```

### [57.和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

`double pointer`

因为是递增序列 搜索优先考虑双指针/二分

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int l = 0, r = nums.length -1 ;
        while (l < r) {
            if (nums[l] + nums[r] < target) {
                l++;
            } else if (nums[l] + nums[r] > target) {
                r--;
            } else return new int[]{nums[l], nums[r]};
        }
        return new int[0];
    }
}
```

### [58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

字符串处理

注意点 匹配多个空格时可以用.split("//s+") 也可以循环内判断

```java
"a     dog" -> 中间有 4 个空格 使用split(" ") 的输出结果为
a	1
	0
	0
	0
	0
dog	3
```

```java
class Solution {
    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        String[] str = s.trim().split(" ");
        for (int i = str.length - 1; i >= 0; i--) {
            if ((str[i]).equals("")) continue;
            sb.append(str[i]).append(" ");
        }
        return sb.toString().trim();
    }
}
```

## :melon:位运算 4

### [15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

`每一位与掩码 & 如果不为0 说明该位为1 时间复杂度 O（1） 执行32次`

```java
class Solution {
    public int hammingWeight(int n) {
		int num = 0;
        int mask = 1;
        for (int i = 0; i < 32; i++) {
			if ((n & mask) != 0) {
                num ++;
            }
            mask <<= 1;
        }
    }
}
```

`利用n & n-1 将最低位的1变为零 每执行一次操作num ++直到n变为0 执行1的个数次`

```java
class Solution {
	public int hammingWeight(int n) {
        int num = 0;
        while(n != 0) {
            num++;
            n &= n - 1;
        }
    }
}
```

### [56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

<u>要求O(N) TIME O(1) SPACE</u>

**异或满足交换律**，第一步异或，相同的数其实都抵消了，剩下两个不同的数。这两个数异或结果肯定有某一位为1，不然都是0的话就是相同数。找到这个位，不同的两个数一个在此位为0，另一个为1。按此位将所有数分成两组，分开后各自异或，相同的两个数异或肯定为0（而且分开的时候，两个数必为一组）。剩下的每组里就是我门要找的数。



如果 数组中只有一个出现一次的数 那么全部异或以后 结果为 该数

所以核心点就是 将这两个数 划分到两个子数组中  然后分别异或 就能分别得到

> 如何 划分两个子数组

通过异或数组运算 可以得到 x ^ y 然后设掩码m = 1 从低位依次向高位进行 & 运算 从而判断 当前位是1 还是0 如果是1 说明x 和 y 在该位异或运算的结果为1 则 x 和 y在该位不同（一个 为 0 一个 为 1）所以 找到第一个异或为1 的位 就可以再对数组进行& 划分依据就是 该位为0的 作为一个数组 为1的作为一个数组 因为 x 和 y 就是依次划分开的 这样 分别进行异或就可以得到x 和 y 

```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        int ret = 0;
        for (int n : nums) {
            ret ^= n;
        }
        int div = 1;
        while ((div & ret) == 0) {
            div <<= 1;
        }
        int a = 0, b = 0;
        for (int n : nums) {
            if ((div & n) != 0) {
                a ^= n;
            } else {
                b ^= n;
            }
        }
        return new int[]{a, b};
    }
}
```

### [56 - II. 数组中数字出现的次数 II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

`hashMap`

```java
class Solution {
    public int singleNumber(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue()== 1) return entry.getKey();
        }
        return -1;
    }
}
```

`位运算`

解决数组中出现m次数字的通用做法 通过更改取余数m即可实现 效率不如状态机

+ 建立一个长度为 32 的数组 counts ，记录所有数字的各二进制位的 11 的出现次数。
+ 将 counts 各元素对 3 求余，则结果为 “只出现一次的数字” 的各二进制位。
+ 利用 左移操作 和 或运算 ，可将 counts 数组中各二进位的值恢复到数字 res 上（循环区间是 i in [0, 31]  i∈[0,31] ）。

```java
public int singleNumber(int[] nums) {
        int[] counts = new int[32];
        for(int num : nums) {
            for(int j = 0; j < 32; j++) {
                counts[j] += num & 1;
                num >>>= 1;
            }
        }
        int res = 0, m = 3;
        for(int i = 0; i < 32; i++) {
            res <<= 1;
            res |= counts[31 - i] % m;
        }
        return res;
    }
```



`状态机`(不懂)

https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/solution/mian-shi-ti-56-ii-shu-zu-zhong-shu-zi-chu-xian-d-4/

```java
class Solution {
    public int singleNumber(int[] nums) {
        int ones = 0, twos = 0;
        for(int num : nums){
            ones = ones ^ num & ~twos;
            twos = twos ^ num & ~ones;
        }
        return ones;
    }
}
```

### [65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

+ 相加 => 异或
+ 进位 => 与操作再左移一位

即 对两个数二进制每一位 都进行上面两个操作 直到进位数为0

补码存储 CPU只有加法运算器 所以同样适用于减法

`recursion`

```java
class Solution {
    public int add(int a, int b) {
        if (b == 0) return a;
        return add(a ^ b, (a & b) << 1);
    }
}
```

`iteration`

```java
class Solution {
    public int add(int a, int b) {
        int c = 0;
        while (b != 0) {
            c = (a & b) << 1;
            a = a ^ b;
            b = c;
        }
        return a;
    }
}
```

## :grapes:数学 8

### [14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

切分规则：
最优： 3 。把绳子尽可能切为多个长度为 3 的片段，留下的最后一段绳子的长度可能为 0,1,2三种情况。
次优： 2 。若最后一段绳子长度为 2 ；则保留，不再拆为 1+1 。
最差： 1 。若最后一段绳子长度为 1 ；则应把一份3+1 替换为2+2，因为 2 * 2  > 3 * 1

```java
class Solution {
    public int cuttingRope(int n) {
        if (n <= 3) return n - 1;
        int a = n / 3, b = n % 3;
        if (b == 0) return (int)Math.pow(3, a);
        if (b == 1) return (int)Math.pow(3, a - 1) * 4;
        return (int)Math.pow(3, a) * 2;
    }
}
```

### [14-II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

核心问题就是a的b次方取余操作， 因为 a的b次方可能会溢出

1. 循环取余 每次保存× a % p的操作 O(N) time
2. 快速幂取余 
   + a 为 偶数时， x 的 a 次幂 取余 p = （a 的 平方 取余 p）的 a / 2 的幂 再取余p
   + a 为奇数时， x 的 a 次幂 取余 p = （a * a 的 平方 取余 p) 的 a / 2的 幂 再 取余p

细节问题

+ rem 和 x 取 long
+ 类型转换时 注意加括号 整体转换

```java
class Solution {
    public int cuttingRope(int n) {
        if(n <= 3) return n - 1;
        int b = n % 3, p = 1000000007;
        long rem = 1, x = 3;
        for(int a = n / 3 - 1; a > 0; a /= 2) {
            if(a % 2 == 1) rem = (rem * x) % p;
            x = (x * x) % p;
        }
        if(b == 0) return (int)(rem * 3 % p);
        if(b == 1) return (int)(rem * 4 % p);
        return (int)(rem * 6 % p);
	}
}
```

### [39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

`hash统计` `排序取中值`

`摩尔投票法`

 核心理念为 **票数正负抵消** O(N) time O(1) space

```java
class Solution {
    public int majorityElement(int[] nums) {
        int x = 0, vote = 0;
        for (int num : nums) {
            if (vote == 0) x = num;
            vote += x == num ? 1 : -1;
        }
        return x;
    }
}
```

### [43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

`统计每一位1出现的次数 累加`

从最低位作为curr，左边为高位，右边为低位  同时记录digit 作为位数

初始化

```java
int high = n / 10;
int curr = n % 10;
int low = 0;
int digit = 1;
```

核心规律

```java
if (curr == 0) 当前位存在的1的个数 = high * digit
else if (curr == 1) 当前位存在的1的个数 = high * digit + low + 1
else 当前位存在的1的个数 = (high + 1) * digit
```

迭代终止条件

```java
while (high != 0 || curr != 0) //即高位high 和 当前为curr 同时为0时 termination
```

变量状态更新

```java
low += curr * digit //相当于low + low左边位置的数(curr * digit)
curr = hight % 10 //当前位左移一位 即 当前高位的最后一位
high /= 10 //高位左移一位
digit *= 10 //进制位 进位操作
```

`code`

```java
class Solution {
    public int countDigitOne(int n) {
        int high = n / 10, curr = n % 10, low = 0;
        int digit = 1, res = 0;
        while (high != 0 || curr != 0) {
            //计算该层res
			if (curr == 0) res += high * digit;
            else if (curr == 1) res += high * digit + low + 1;
            else res += (high + 1) * digit;
            //更新状态 注意b
            low += curr * digit;
            curr = high % 10;
            high /= 10;
            digit *= 10;
        }
        return res;
    }
}
```




### [44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

`Mathematical laws`

1.  确定 n 所在 数字 的 位数 ，记为 digit ；		count = digit * start * 9
2.  确定 n 所在的 数字 ，记为 num ；     	 num = start + (n - 1) / digit
3.  确定 n 是 num 中的哪一数位，并返回结果。			     (n -1) % digit

```java
class Solution {
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1, count = 9;
        //确定所在的位数
        while (n > count) {
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        //确定所在的数字num 
        long num = start + (n - 1) / digit;
        //确定所在数字第几位
        return Long.toString(num).charAt((n - 1) % digit) - '0';

    }
}
```
###  [57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

`滑动窗口` `double pointer`

未使用求和公式，可以解决任意的递增整数序列 的通用解法

```java
class Solution {
    public int[][] findContinuousSequence(int target) {
        int i = 1; // 滑动窗口的左边界
        int j = 1; // 滑动窗口的右边界
        int sum = 0; // 滑动窗口中数字的和
        List<int[]> res = new ArrayList<>();

        while (i <= target / 2) {
            if (sum < target) {
                // 右边界向右移动
                sum += j;
                j++;
            } else if (sum > target) {
                // 左边界向右移动
                sum -= i;
                i++;
            } else {
                // 记录结果
                int[] arr = new int[j - i];
                for (int k = i; k < j; k++) {
                    arr[k-i] = k;
                }
                res.add(arr);
                // 左边界向右移动
                sum -= i;
                i++;
            }
        }
        return res.toArray(new int[res.size()][]);
	}
}
```

### [62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

`数学解法` 约瑟夫环问题

因为数组是自然数有序，所以ans = index(ans)，所以我们只要锁定ans从最后的index（也就是0）反推ans在每一次变更后的index（通过补上移除的数字的位置），最终获得ans最初的index，也就拿到ans的值了。

**(当前index + m) % 上一轮剩余数字的个数**

```java
class Solution {
    public int lastRemaing(int n, int m) {
        int ans = 0;
        //最后一轮剩下2个人，所以从2开始反推
        for (int i = 2; i <= n; i++) {
            ans = (ans + m) % i;
        }
        return ans;
    }
}
```


### [66. 构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

`实质是dp`

<img src="C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210515103855370.png" alt="image-20210515103855370" style="zoom:150%;" />

下三角 状态转移方程 b[i] = b[i - 1] * a[i]

上三角 状态转移方程 b[i] = a[i + 1] * .......*a[a.length - 1] (使用temp保存连乘的中间值)

O(N) time 暴力会超时

```java
class Solution {
    public int[] constructArr(int[] a) {
        if (a == null || a.length == 0) return a;
        int[] b = new int[a.length];
        b[0] = 1;
        for (int i = 1; i < a.length; i++) {
            b[i] = b[i - 1] * a[i - 1];
        }
        int temp = 1;
        for (int i = a.length - 2; i >= 0; i--) {
            temp *= a[i + 1];
            b[i] *= temp;
        }
        return b;
    }
}
```

## :lemon:模拟

### [29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

模拟法，设定边界tar

```java
class Solution {
    public int[] spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) return new int[0];
        int l = 0, r = matrix[0].length - 1, t = 0, b = matrix.length - 1;
        int tar = (r + 1) * (b + 1);
        int cnt = 0;
        int[] res = new int[tar];
        while (tar >= 1) {
            //left to right t++
            for (int i = l; i <= r && tar-- >= 1; i++) res[cnt++] = matrix[t][i];
            t++;
            //top to bottom r--
            for (int i = t; i <= b && tar-- >= 1; i++) res[cnt++] = matrix[i][r];
            r--;
            //right to left b--
            for (int i = r; i >= l && tar-- >= 1; i--) res[cnt++] = matrix[b][i];
            b--;
            //bottom to top l++
            for (int i = b; i >= t && tar-- >= 1; i--) res[cnt++] = matrix[i][l];
            l++;
        }
        return res;
    }
}
```

### [31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

`辅助栈`模拟弹出序列

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> stack = new LinkedList<>();
        int idx = 0;
        for (int num : pushed) {
            stack.push(num);
            while (!stack.isEmpty() && stack.peek() == popped[idx]) {
                stack.pop();
                idx++;
            }
        }
        return stack.isEmpty();
    }
}
```

























































 
