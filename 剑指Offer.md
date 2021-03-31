# 剑指Offer(Frequency desc)

## 03.数组中重复的数字

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

## 24.反转链表

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

## 22.链表中倒数第k个节点

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

## 09.用两个栈实现队列

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
        if (!B.isEmpty()) return B.poll();
        if (A.isEmpty()) return -1;
        while (!A.isEmpty()) B.push(A.poll());
        return B.poll();
    }
    	
}
```

## 38.字符串的排列

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

## 42.连续子数组的最大和

`dp` O(N) time O(1) space

状态定义： dp[i] 代表以元素nums[i] 为结尾的连续子数组最大和

状态转移方程：

dp = dp[i - 1] + nums[i], dp[i - 1] > 0 

​     = nums[i], dp[i - 1] <= 0

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



## 07.重建二叉树

`recursion--divide & conquer`]

O(N) time / O(N) space (best O(logN) space---full b tree)

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

## 25.合并两个排序的链表

`recursion`

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

## 20.表示数值的字符串

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

## 51.数组中的逆序对

逆序对：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。

`merge` O(nlog(n))

```java
class Solution {
 	public int reversePairs(int[]nums) {
        if (nums == null || nums.length == 0) return 0;
        return merge(nums,0, nums.length - 1);
    }   
    public int merge(int[] nums, int left, int right) {
        if (right <= left) return 0;
        int mid = (right + left) >> 1;
        int cnt = merge(nums,left, mid) + merge(nums, mid + 1, right);
        int i = left, j = mid + 1, p = mid + 1,k = 0;
        int[] temp = new int[right - left + 1];
        while (i <= mid) {
            while (p <= right && nums[i] > nums[p]) p++;
            cnt += p - (mid + 1);
            while (j <= right && nums[j] < nums[i]) temp[k++] = nums[j++];
            temp[k++] = nums[i++];
        }
        while (j <= right) temp[k++] = nums[j++];
        System.arraycopy(temp,0,nums,left,temp.length);
        return cnt;
    }
    
}
```

## 29.顺时针打印矩阵（螺旋矩阵）

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

## 04.二位数组中的查找

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

## 48.最长不含重复字符的子字符串

`dp`

**状态定义**

设动态规划列表 dp*d**p* ，dp[j]*d**p*[*j*] 代表以字符 s[j]*s*[*j*] 为结尾的 “最长不重复子字符串” 的长度。

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

## 10.II 青蛙跳台阶问题

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

## [40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

`quick select`

思想类似于快排，但是分区之后不排序

- k == m 直接返回左数组
- k < m 返回长度为k的左子数组
- k > m 返回左数组 + (k - m)继续快速选择右数组

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k >= arr.length) return arr;
        return quickSort(arr, k, 0, arr.length - 1);
    }
    private int[] quickSort(int[] arr, int k, int begin, int end) {
        int i = begin, j = end;
        while (i < j) {
            //此处两个while的顺序不能调换
            while (i < j && arr[j] >= arr[begin]) j--;
            while (i < j && arr[i] <= arr[begin]) i++;
            swap(arr, i, j);
        }
        swap(arr, i, begin);
        if (i > k) return quickSort(arr, k, begin, i - 1);
        if (i < k) return quickSort(arr, k ,i + 1, end);
        return Arrays.copyOf(arr, k);
    }
    void swap(int[] arr, int a, int b) {
        int temp = arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
    }
}
```

## [11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

`二分法`

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

## [53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

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

## [10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

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

## [63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

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

## [57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

`滑动窗口`

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

## [06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

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

## [52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

`double pointer`

**当节点为null时 返回另一个节点的头节点**

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A = headA, B = headB;
        while (A != B) {
            A = A != null ? A.next : headB;
            B = B != null ? B.next : headA;
        }
        return A;
    }
}
```

## [ 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

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

## [55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

`recursion`

```java
class Solution {
 	public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
	}   
}
```

## [26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

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

## [13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

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

​	设x的数位和为s1，x+1的数位和为s2

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

## [56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

<u>要求O(N) TIME O(1) SPACE</u>

**异或满足交换律**，第一步异或，相同的数其实都抵消了，剩下两个不同的数。这两个数异或结果肯定有某一位为1，不然都是0的话就是相同数。找到这个位，不同的两个数一个在此位为0，另一个为1。按此位将所有数分成两组，分开后各自异或，相同的两个数异或肯定为0（而且分开的时候，两个数必为一组）。剩下的每组里就是我门要找的数。

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

## [45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

此题求拼接起来的最小数字，本质上是一个排序问题。设数组 numsnums 中任意两数字的字符串为 xx 和 yy ，则规定 排序判断规则 为：

若拼接字符串 x + y > y + xx+y>y+x ，则 xx “大于” yy ；
反之，若 x + y < y + xx+y<y+x ，则 xx “小于” yy ；

x* “小于” y*y* 代表：排序完成后，数组中 x*x* 应在 y*y* 左边；“大于” 则反之。

**O(N log N)** **time**  快排average   **O(N) space** String数组

```java
class Solution {
    public String minNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for (int i = 0; i < nums.length; i++) strs[i] = String.valueOf(nums[i]);
        quickSort(strs, 0, strs.length - 1);
        StringBuilder res = new StringBuilder();
        for (String s : strs) res.append(s);
        return res.toString();
    }
    void quickSort(String[] strs, int l, int r) {
        if (r <= l) return;
        int i = l, j = r;
        String temp = strs[i];
        while (i < j) {
            //以l作为pivot
            while (i < j && (strs[j] + strs[l]).compareTo(strs[l] + strs[j]) >= 0) j--;
            while (i < j && (strs[i] + strs[l]).compareTo(strs[l] + strs[i]) <= 0) i++;
            temp = strs[i];
            strs[i] = strs[j];
            strs[j] = temp;
        }
        strs[i] = strs[l];
        strs[l] = temp;
        quickSort(strs, l, i - 1);
        quickSort(strs, i + 1, r);
    }
}
```

## [12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

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

## [05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

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

## [46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

`dp` 节省了空间

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

## [58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

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

