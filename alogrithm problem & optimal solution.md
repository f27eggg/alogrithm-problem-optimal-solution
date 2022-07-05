## two sum

`brute force`  `O(n^2)`

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        for (int i = 0; i < nums.length - 1; ++ i) {
            for (int j = i + 1; j < nums.length; ++j) {
                if ((nums[i] + nums[j]) == target) {
                    return new int[]{i,j};
                }
            }
        }
        return new int[2];
    }
}
```

`hash`  `O(n)`

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])){
                return new int[]{i,map.get(target - nums[i])};
            }
            map.put(nums[i],i);
        }
        return new int[0];
    }
}
```

## 3 sum

`double pointer`

```java
public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(nums);
    for (int i = 0; i + 2 < nums.length; i++) {
        if (nums[i] > 0) break;                             // if sorted array the first element > 0 then result is checked
        if (i > 0 && nums[i] == nums[i - 1]) {              // skip same result
            continue;
        }
        int j = i + 1, k = nums.length - 1;  
        int target = -nums[i];
        while (j < k) {
            if (nums[j] + nums[k] == target) {
                res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                j++;
                k--;
                while (j < k && nums[j] == nums[j - 1]) j++;  // skip same result
                while (j < k && nums[k] == nums[k + 1]) k--;  // skip same result
            } else if (nums[j] + nums[k] > target) {
                k--;
            } else {
                j++;
            }
        }
    }
    return res;
}
```

## 4 sum

`double pointer`

>  p.s. 还可以优化 有剪枝操作

二刷记一个细节 关于 nums = [2,2,2,2,2] target = 8

两层循环的 continue条件问题 虽然 j 一定是 > 0的 但是 也要保证 j > i + 1，这个条件的意思就是进入j循环的第一个数 不参与去重，（i 也是同理 但同时保证数组不会越界）

这样进入到k 和 p时 i 和 j 分别为 2 和 2 如果不加这个条件 那么在第一次j = 2时就 直接一直continue出循环了

```java
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        if (nums.length < 3) return new ArrayList<>();
        Arrays.sort(nums);
        List<List<Integer>> list = new ArrayList<>();
        for (int i = 0; i + 3 < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            for (int j = i + 1; j + 2 < nums.length; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                 int k = j + 1, p = nums.length - 1;
                 int curTar = target - nums[i] - nums[j];
                 while (k < p) {
                     if (nums[k] + nums[p] == curTar) {
                         list.add(Arrays.asList(nums[i], nums[j], nums[k], nums[p]));
                         k++;p--;
                         while (k < p && nums[k] == nums[k - 1]) k++;
                         while (k < p && nums[p] == nums[p + 1]) p--;
                     } else if (nums[k] + nums[p] > curTar) {
                         p--;
                     } else k++;
                 }
            }
        }
        return list;
    }
}
```

## add two numbers

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0);
        ListNode result = head;
        int carry = 0;
        while (l1 != null || l2 != null || carry > 0) {
            int res = (l1 != null ? l1.val : 0) + (l2 != null ? l2.val : 0) + carry;
            result.next = new ListNode(res % 10);
            carry = res / 10;
            l1 = (l1 == null ? l1 : l1.next);
            l2 = (l2 == null ? l2 : l2.next);
            result = result.next;
        }
        return head.next;
    }
}
```

## reverse integer

1. 反转通过取余获得尾数作为高位
2. 每次计算当前数 再回退 比较上一次的结果 判断是否相等得到是否溢出

Integer.MAX_VALUE + 1 = Integer.MIN_VALUE

```java
class Solution {
    public int reverse(int x) {
        int res = 0;
        while (x != 0) {
            int tail = x % 10;
            int newRes = res * 10 + tail;
            if ((newRes - tail) / 10 != res) return 0;//校验是否溢出，如果溢出则不相等
            res = newRes;
            x /= 10;
        }
        return res;
    }
}
```

## reverse linked list

`iteration` save curr.next as tempNext and make curr point at prev then move forward O(N) O(1)

```java
class Solution {
	public ListNode reversList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode tempNext = curr.next;
            curr.next = prev;
            prev = curr;
            curr = tempNext;
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

## rotate list

通过快慢指针找到要拼接的节点，然后进行拼接。

旋转=没旋转的情况 1) null 2) k % len == 0

```java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return head;
        int len = 0;
        ListNode fast = head;
        while (fast != null) {
            fast = fast.next;
            len++;
        }
        if (k % len == 0) return head;
        else {
            int t = 0;
            fast = head;
            ListNode slow = head;
            ListNode prev = null;
            while (fast != null) {
                if (t++ > (k % len)) slow = slow.next;
                prev = fast;
                fast = fast.next;
            }
            ListNode newHead = slow.next;
            slow.next = null;
            prev.next = head;
            return newHead;
            
        }
    }
}
```

## reorder list

由中点切分两个链表，将第二个链表倒置，然后依次合并两个链表

```java
public void reorderList(ListNode head) {
    if (head == null || head.next == null || head.next.next == null) {
        return;
    }
    //找中点，链表分成两个
    ListNode slow = head;
    ListNode fast = head;
    while (fast.next != null && fast.next.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }

    ListNode newHead = slow.next;
    slow.next = null;
    
    //第二个链表倒置
    newHead = reverseList(newHead);
    
    //链表节点依次连接
    while (newHead != null) {
        ListNode temp = newHead.next;
        newHead.next = head.next;
        head.next = newHead;
        head = newHead.next;
        newHead = temp;
    }

}

private ListNode reverseList(ListNode head) {
    if (head == null) {
        return null;
    }
    ListNode tail = head;
    head = head.next;

    tail.next = null;

    while (head != null) {
        ListNode temp = head.next;
        head.next = tail;
        tail = head;
        head = temp;
    }

    return tail;
}
```

## palindrome linked list

快慢指针找到中点 反转后半部分链表，与头链表依次比较

```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        if (head == null) return true;
        ListNode fast = head, slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode temp = slow.next;
        slow.next = null;
        slow = reverseList(temp);
        while (slow != null) {
            if (head.val != slow.val) return false;
            head = head.next;
            slow = slow.next;
        }
        return true;
    }
    ListNode reverseList(ListNode head) {
        if (head == null) return head;
        ListNode curr = head, prev = null;
        while (curr != null) {
            ListNode x = curr.next;
            curr.next = prev;
            prev = curr;
            curr= x;
        } 
        return prev;
    }
}
```

## reverse nodes in k group

```java
public ListNode reverseKGroup(ListNode head, int k) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;

    ListNode pre = dummy;
    ListNode end = dummy;

    while (end.next != null) {
        for (int i = 0; i < k && end != null; i++) end = end.next;
        if (end == null) break;
        ListNode start = pre.next;
        ListNode next = end.next;
        end.next = null;
        pre.next = reverse(start);
        start.next = next;
        pre = start;

        end = pre;
    }
    return dummy.next;
}

private ListNode reverse(ListNode head) {
    ListNode pre = null;
    ListNode curr = head;
    while (curr != null) {
        ListNode next = curr.next;
        curr.next = pre;
        pre = curr;
        curr = next;
    }
    return pre;
}
```

## merge k sorted lists

`merge`

```java
public static ListNode mergeKLists(ListNode[] lists){
    return partion(lists,0,lists.length-1);
}

public static ListNode partion(ListNode[] lists,int s,int e){
    if(s==e)  return lists[s];
    if(s<e){
        int q=(s+e)/2;
        ListNode l1=partion(lists,s,q);
        ListNode l2=partion(lists,q+1,e);
        return merge(l1,l2);
    }else
        return null;
}

//This function is from Merge Two Sorted Lists.
public static ListNode merge(ListNode l1,ListNode l2){
    if(l1==null) return l2;
    if(l2==null) return l1;
    if(l1.val<l2.val){
        l1.next=merge(l1.next,l2);
        return l1;
    }else{
        l2.next=merge(l1,l2.next);
        return l2;
    }
}
```

## sort list

```java
public class Solution {
  
  public ListNode sortList(ListNode head) {
    if (head == null || head.next == null)
      return head;
        
    // step 1. cut the list to two halves
    ListNode prev = null, slow = head, fast = head;
    
    while (fast != null && fast.next != null) {
      prev = slow;
      slow = slow.next;
      fast = fast.next.next;
    }
    
    prev.next = null;
    
    // step 2. sort each half
    ListNode l1 = sortList(head);
    ListNode l2 = sortList(slow);
    
    // step 3. merge l1 and l2
    return merge(l1, l2);
  }
  
  ListNode merge(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0), p = d;
    
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
    
    p.next = l1 == null ? l2 : l1;
    return l.next;
  }

}
```

## delete node in a linked list

由于无法得知删除节点的prev node，所以将该node作为prev node

- 给定的节点为非末尾节点并且一定是链表中的一个有效节点。

```java
class Solution {
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}
```

remove linked lsit elements

`iteration` dummy node

```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        //创建一个虚拟头结点
        ListNode dummyNode=new ListNode(val-1);
        dummyNode.next=head;
        ListNode prev=dummyNode;
        //确保当前结点后还有结点
        while(prev.next!=null){
            if(prev.next.val==val){
                prev.next=prev.next.next;
            }else{
                prev=prev.next;
            }
        }
        return dummyNode.next;
    }
}
```

`*recursion`

```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
       if(head==null)
           return null;
        head.next=removeElements(head.next,val);
        if(head.val==val){
            return head.next;
        }else{
            return head;
        }
    }
}
```



## linked list cycle

`hash set` O(N) O(N)

```java
class Solution {
    public boolean hasCycle(ListNode head) {
		Set<ListNode> set = new HashSet<>();
        while(!set.add(head)) return true;
        head = head.next;
    }
    return false;
}
```

`fast & slow pointer` O(n) O(1)  **21/4/15 15:00(UTC-8) 快手一面**

```java
class Solution {
    public boolean hasCycle (ListNode head) {
		if (head == null || head.next == null) return false;
        ListNode fast = head.next;
        ListNode slow = head;
        while (fast != slow) {
            if (fast == null || fast.next == null) return false;
            fast = fast.next.next;
            slow = slow.next;
        }
        return true;
    }
}
```

## linked list cycle ii

fast & slow pointer

类比与i题 如果fast 和 slow 都指向head 第一次相遇时 fast 和 slow 分别走了 2nc 和 nc 的长度（c为周长）

此时将fast 置为head 当 slow 走到重合链表初（a步） 则 fast = a slow = a  + nc 两指针重合

```java
public ListNode detectCycle(ListNode head) {
        ListNode fast = head, slow = head;
        while (true) {
            if (fast == null || fast.next == null) return null;
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) break;
        }
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return fast;
    }

```

## find the duplicate number

`cycle linkedList`

建立一个 idx -> nums[idx] 的映射关系 如果有重复的数字 则一定会有多对一的映射 即存在闭环

1. 如何递推 fast = nums[nums[fast]] slow = nums[slow]
2. 如何找到重复的数字 找到形成环的起点

```java
class Solution {
    public int findDuplicate(int[] nums) {
        //初始化很妙 相当于直接跳过第一次 这样可以在while里写终止条件 而不用在while里break
        int fast = nums[nums[0]];
        int slow = nums[0];
        while (fast != slow) {
            fast = nums[nums[fast]];
            slow = nums[slow];
        }
        fast = 0;
        while(fast != slow){
            fast = nums[fast];
            slow = nums[slow];
        }
        return slow;
    }
}
```



## remove nth node from end of list

`double pointer` fast and slow pointer stay n gap 

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode start = new ListNode(0);
        ListNode slow = start, fast = start;
        start.next = head;
        for (int i = 0; i < n + 1; i++) fast = fast.next;
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return start.next;
    }
}
```



## merge two sorted lists

`recursion`

```java
class Solution {
	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        else if (l2 == null) return l1;
        else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
			l2.next = mergeTwoSorts(l1, l2.next);
            return l2;
        }
    }
}
```

`iteration`

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode p, dummy = new ListNode(0);
        p = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                p.next = l1;
                l1 = l1.next;
            } else {
                p.next = l2;
                l2 = l2.next;
            }
            p = p.next;
        }
        p.next = (l1 == null) ? l2 : l1;
        return dummy.next;
    }
}
```



## remove duplicates from sorted array

`fast & slow pointer`     要求删除重复元素，实际上就是将不重复的元素移到数组的左侧。

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int p = 0, q = 1;
        while (q < nums.length) {
			if (nums[p] == nums[q]) q++;
            else {
				nums[p + 1] = nums[q];
                p++; q++;
            }
        }
        return p + 1;
    }
}
```

## remove element 

`double point`

类似于上题 通过双指针 进行swap

```java
class Solution {
    public int removeElement(int[] nums, int val) {
        int j = nums.length - 1;
        for (int i = 0; i <= j; i++) {
            if (nums[i] == val) {
                swap(nums, i--, j--);
            }
        }
        return ++j;
    }
    void swap(int[] nums, int a, int b) {
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    }
}
```

`保留逻辑`

```java
class Solution {
    public int removeElement(int[] nums, int val) {
		int idx = 0;
        for (int num : nums) {
            if (num != val) nums[idx++] = num;
        }
         return idx;
    }
}

```

## roman to integer

保留前一位 与当前位比较

如果小于 做减法

如果大于 做加法

```java
import java.util.*;

class Solution {
    public int romanToInt(String s) {
        int sum = 0;
        int preNum = getValue(s.charAt(0));
        for(int i = 1;i < s.length(); i ++) {
            int num = getValue(s.charAt(i));
            if(preNum < num) {
                sum -= preNum;
            } else {
                sum += preNum;
            }
            preNum = num;
        }
        sum += preNum;
        return sum;
    }
    
    private int getValue(char ch) {
        switch(ch) {
            case 'I': return 1;
            case 'V': return 5;
            case 'X': return 10;
            case 'L': return 50;
            case 'C': return 100;
            case 'D': return 500;
            case 'M': return 1000;
            default: return 0;
        }
    }
}
```



## merge sorted array

`double pointer` O(m + n) time O(m) space  从前往后

```java
class Solution {
    public void merge (int[] nums1, int m, int[] nums2, int n) {
		int[] copy = new int[m];
        System.arraycopy(nums1, 0, copy, 0, m);
        int i = 0, j = 0, k = 0;
        while (i < copy.length && j < nums2.length) {
            nums1[k++] = copy[i] < nums2[j] ? copy[i++] : nums2[j++];
        }
        //add the rest element
        while (i < copy.length) nums1[k++] = copy[i++];
        while (j < nums2.length) nums1[k++] = nums2[j++];
    }
}
```

`double pointer` O(m + n) time O(1) space 从后往前

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        while (i >= 0 && j >= 0) {
            nums1[k--] = nums1[i] > nums2[j] ? nums1[i--] : nums2[j--];
        }
        System.arraycopy(nums2, 0, nums1, 0, j + 1);
    }
}
```

## rotate array

`simple`

```java
class Solution {
    public void rotate(int[] nums, int k) {
         k %= nums.length;
        int[] temp = new int[nums.length];
        for (int i = k; i < nums.length; i++) {
            temp[i] = nums[i - k];
        }
        for (int i = 0; i < k; i++) {
            temp[i] = nums[nums.length - k + i];
        }
        System.arraycopy(temp, 0, nums, 0, temp.length);
    }
}
```



`extra space` O(n) O(n)

```java
class Solution {
	public void retate (int[] nums, int k) {
        int[] temp = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            temp[(i + k) % nums.length] = nums[i];
        }
        System.arraycopy(temp, 0, nums, 0, temp.length);
    }
}
```

`rotate array `

```java
class Solution {
    public void rotate(int[] nums, int k) {
        k %= nums.length; //virtal cause k may bigger than the size of the array
        reverse(0, nums.length - 1, nums);
        reverse(0, k - 1, nums);
        reverse(k, nums.length - 1, nums);
    }
    public void reverse (int l, int r, int[] nums) {
        while (l < r) {
            int temp = nums[l];
            nums[l++] = nums[r];
            nums[r--] = temp;
        }
    }
}
```

## valid parentheses

`stack`

```java
public class Solution {
    public boolean isValid(String s) {
        Deque<Character> stack = new ArrayDeque<>();
      	for (char c : s.toCharArray()) {
            if (c == '(') {
                stack.push(')');
            } else if (c == '{') {
                stack.push('}');
            } else if (c == '[') {
                stack.push(']');
            } else if (stack.isEmpty() || stack.pop() != c) {
                return false;
            }
        }
        return stack.isEmpty();
    }
}
```

## Largest rectangle in histogram

`stack O(n)`

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int len =  heights.length;
        int max_area = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i <= len;) {
            int h = (len == i ? 0 : heights[i]);
            if (stack.isEmpty() || h >= heights[stack.peek()]) {
                stack.push(i);
                i++;
            } else {
                int currHeight = heights[stack.pop()];
                int rightBoundary = i - 1;
                //左边界需要考虑是否在最左边
                int leftBoundary = stack.isEmpty() ? 0 : stack.peek() + 1;
                int width = rightBoundary - leftBoundary + 1;
                max_area = Math.max(max_area, currHeight * width);
            }
        }
        return max_area;
    }
}
```

## sliding window maximum

`ArrayDeque` O(n) time 用双向队列的思想

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        if (nums == null || n == 0) return nums;
        int[] res = new int[n - k + 1];
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            //remove element out of range k
            if (!dq.isEmpty() && dq.peek() < i - k + 1) {
                dq.poll();
            }
            //remove smaller element in k range as they are useless
            while (!dq.isEmpty() && nums[i] >= nums[dq.peekLast()]) {
                dq.pollLast();
            }
            dq.a(i);
            if (i - k + 1 >= 0) {
                res[i - k + 1] = nums[dq.peek()];
            }
        }
        return res;
    }
}
```



## Pre Order Traverse

`iteration`

```java
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
   	TreeNode p = root;
    while (p != null || !stack.isEmpty()) {
        if (p != null) {
            stack.push(p);
            list.add(p.val); //Add before going to children
            p = p.left;
        } else {
            TreeNode node = stack.pop();
            p = node.right;
        }
    }
    return list;
}
```

## In Order Traverse

`Iteration`

```java
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> list = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode p = root;
    while (p != null || !stack.isEmpty()) {
        if ( p != null) {
            stack.push(p);
            p = p.left;
        } else {
            TreeNode node = stack.pop();
            list.add(p.val); //Add after all left children
            p = node.right;
        }
    }
    return list;
}
```

## Post Order Traverse

`Iteration`

```java
public List<Integer> postorderTraversal(TreeNode root) {
    LinkedList<Integer> list = new LinkedList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode p = root;
    while (p != null || !stack.isEmpty()) {
        if (p != null) {
            stack.push(p);
            list.addFirst(p.val); //Reverse the process of preorder
            p = p.right;		  //Reverse the process of preorder
        } else {
            TreeNode node = stack.pop();
            p = node.left;		  //Reverse the process of preorder
        }
    }
    return list;
}
```

## move zero

`using insert index O(N)`

```java
//shift non-zero values as far forward as possible
//fill remaining space with zeros

public void moveZeroes(int[] nums) {
    if (nums == null || nums.length == 0) return;
    int insertPos = 0;
    for (int  num: nums) {
        if (num != 0) nums[insertPos++] = num;
    }
    while (insertPos < nums.length) {
        nums[insertPos++] = 0;
    }
}
```

`double pointer O(n)`

```java
public void moveZeroes(int[] nums) {
    int j = 0;
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] != 0){
            //sawp nums[i] & nums[j] and i,j++
        	int tmp = nums[j];
            nums[j] = nums[i];
            nums[i] = tmp;
            j++;
        }
    }
}
```

`double pointer without using sawp`

```java
public void moveZeroes(int[] nums) {
    int j = 0;
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] != 0) {
            if (i > j) {
                nums[j] = nums[i];
                nums[i] = 0;
            }
            j++;
        }
    }
}
```

## Container with most water

`double pointer`

```java
public int maxArea(int[] height) {
        int max_area = 0;
        for (int i = 0, j = height.length - 1; i < j;) {
            int minHeight = (height[i] < height[j] ?  height[i++] : height[j--]);
            max_area = Math.max(max_area,minHeight * (j - i + 1));
        }
        return max_area;
    }
```

## trapping rain water

`double pointer` O(n) time O(1) space

+ 初始化left指针为0且right指针为size-1
+ while left < right, do:
  + if height[left]  < height[right]
    + if height[left] >= left_max, update left_max
    + else accumulate res +=  left_max - height[left]
    + left  = left + 1
  + else 
    + if height[right] >= right_max, update right_max
    + else accumulate res += right_max - height[right]
    + right = right - 1

```java
public int trap(int[] height) {
	int left = 0, right = height.length - 1;
    int res = 0;
    int left_max = 0, right_max = 0;
    while (left < right) {
        if (height[left] < height[right]) {
			if (height[left] >= left_max) {
                left_max = height[left];
            } else {
                res += left_max - height[left];
            }
        	left++;
        } else {
         	if (height[right] >= right_max) {
                right_max = height[right];
            } else {
                res += right_max - height[right];
            }
            right--;
        }
    }
    return res;
}
```



## Climb stairs

`Dynamic planning using sliding window to solve Fibonacci problem `

```java
public int climbStairs(int n) {
        int p = 0, q = 0, r = 1;
        for (int i = 0; i < n; i++) {
            p = q;
            q = r;
            r = p + q;
        }
        return r;
    }
```

## Invert binary tree

`recursion`

```java
public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode left = root.left, right = root.right;
        root.left = invertTree(right);
        root.right = invertTree(left);
        return root;
    }
```

## validate binary search tree

`recursion O(n) `

```java
public TreeNode isValidBST(TreeNode root) {
    return helper(root, null, null);
}
public boolean helper(TreeNode root, Integer min, Integer max) {
    if (root == null) {
        return true;
    }
    if ((min != null && root.val <= min) || (max != null && root.val >= max)) {
        return false;
    }
    return helper(root.left, min, root.val) && helper(root.right, root.val, max);
}
```

`iteration O(n) `

```java
public boolean isValidBST(TreeNode root) {
    if (root == null) return true;
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode p = root;
    TreeNode pre = null;
    while (p != null || !stack.isEmpty()) {
        if (p != null) {
            stack.push(p);
            p = p.left;
        } else {
            TreeNode node = stack.pop();
            if (pre != null && node.val <= pre.val) return false; 
            pre = node;
            p = node.right;
        }
    }
    return true;
}
```

`recursion by inorder only mark pre element  `

```java
Integer pre = null;
public boolean isValidBST(TreeNode root) {
    return helper(root);
}
public boolean helper(TreeNode root) {
    if (root == null) return true;
    if (!helper(root.left)) return false;
    if (pre != null && root.val <= pre) return false;
    pre = root.val;
    return helper(root.right);
}
```

## maximum depth of binray tree

`recursion`

```java
public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
}
```

`DFS`

```java
public int maxDepth(TreeNode root) {
    if(root == null) {
        return 0;
    }
    Stack<TreeNode> stack = new Stack<>();
    Stack<Integer> value = new Stack<>();
    stack.push(root);
    value.push(1);
    int max = 0;
    while(!stack.isEmpty()) {
        TreeNode node = stack.pop();
        int temp = value.pop();
        max = Math.max(temp, max);
        if(node.left != null) {
            stack.push(node.left);
            value.push(temp+1);
        }
        if(node.right != null) {
            stack.push(node.right);
            value.push(temp+1);
        }
    }
    return max;
}
```

`BFS`

```java
public int maxDepth(TreeNode root) {
    if(root == null) {
        return 0;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    int count = 0;
    while(!queue.isEmpty()) {
        int size = queue.size();
        while(size-- > 0) {
            TreeNode node = queue.poll();
            if(node.left != null) {
                queue.offer(node.left);
            }
            if(node.right != null) {
                queue.offer(node.right);
            }
        }
        count++;
    }
    return count;
}
```

## same tree

`recurion`

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        else if (p == null || q == null) return false;
        else if (p.val != q.val) return false;
        else return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
```



## minimum depth of binary tree

`recursion`

```java
public int minDepth(TreeNode root) {
    if (root == null) {
        return 0;
    }
	int left = minDepth(root.left);    
    int right = minDepth(root.right);  
    return (left == 0 || right == 0) ? left + right + 1 : 1 + Math.min(left,right);
}
```

`BFS optimal`

```java
public int minDepth(TreeNode root) {
	if (root == null) return 0;

	Queue<TreeNode> queue = new ArrayDeque<>();
	queue.offer(root);

	int depth = 0;
	while (!queue.isEmpty()) {
		int size = queue.size();
		while (size > 0) {
			TreeNode cur = queue.poll();

			if (cur.left == null && cur.right == null)
				return ++depth;

			if (cur.left != null) queue.offer(cur.left);
			if (cur.right != null) queue.offer(cur.right);

			size--;
		}
		depth++;
	}
	return depth;
}
```

## serialize-and-deserialize-binary-tree

`Serialize --DFS preorder StringBuilder`

`Deserialize -- DFS preorder Deque`

```java
public class Codec {
    private static final String spliter = ",";
    private static final String NN = "X";

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        buildString(root, sb);
        return sb.toString();
    }

    private void buildString(TreeNode node, StringBuilder sb) {
        if (node == null) {
            sb.append(NN).append(spliter);
        } else {
            sb.append(node.val).append(spliter);
            buildString(node.left, sb);
            buildString(node.right,sb);
        }
    }
    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Deque<String> nodes = new LinkedList<>();
        nodes.addAll(Arrays.asList(data.split(spliter)));
        return buildTree(nodes);
    }
    
    private TreeNode buildTree(Deque<String> nodes) {
        String val = nodes.remove();
        if (val.equals(NN)) return null;
        else {
            TreeNode node = new TreeNode(Integer.valueOf(val));
            node.left = buildTree(nodes);
            node.right = buildTree(nodes);
            return node;
        }
    }
}
```

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



## lowest common ancestor of a binary search tree

也可以用二叉树的最近公共祖先解法(通用解法)

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



## Lowest common ancestor of a binary tree

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



## lowest common ancestor of deepest leaves

`recursion`

```java
class Solution {	
	int deepest = 0;
    TreeNode lca;

    public TreeNode lcaDeepestLeaves(TreeNode root) {
        helper(root, 0);
        return lca;
    }

    private int helper(TreeNode node, int depth) {
        deepest = Math.max(deepest, depth);
        if (node == null) {
            return depth;
        }
        int left = helper(node.left, depth + 1);
        int right = helper(node.right, depth + 1);
        if (left == deepest && right == deepest) {
            lca = node;
        }
        return Math.max(left, right);
    }
}
```

## Construct Binary Search Tree from Preorder Traversal

`recursion`

















`思路:我们使用递归的方法，在扫描先序遍历的同时构造出二叉树。我们在递归时维护一个 (lower, upper) 二元组，表示当前位置可以插入的节点的值的上下界。如果此时先序遍历位置的值处于上下界中，就将这个值作为新的节点插入到当前位置，并递归地处理当前位置的左右孩子的两个位置。否则回溯到当前位置的父节点。`

![bla](https://pic.leetcode-cn.com/Figures/1008/recursion2.png)

```java
class Solution {
    int idx = 0;
    int[] preorder;
    int n;

    public TreeNode helper(int lower, int upper) {
        // if all elements from preorder are used
        // then the tree is constructed
        if (idx == n) return null;

        int val = preorder[idx];
        // if the current element 
        // couldn't be placed here to meet BST requirements
        if (val < lower || val > upper) return null;

        // place the current element
        // and recursively construct subtrees
        idx++;
        TreeNode root = new TreeNode(val);
        root.left = helper(lower, val);
        root.right = helper(val, upper);
        return root;
    }

    public TreeNode bstFromPreorder(int[] preorder) {
        this.preorder = preorder;
        n = preorder.length;
        return helper(Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
}
```

## construct binary tree from preorder and inorder traversal



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

## Combinations

`recursion`

```java
class Solution {
    List<Integer> temp = new ArrayList<Integer>();
    List<List<Integer>> ans = new ArrayList<List<Integer>>();

    public List<List<Integer>> combine(int n, int k) {
        dfs(1, n, k);
        return ans;
    }

    public void dfs(int cur, int n, int k) {
        // 剪枝：temp 长度加上区间 [cur, n] 的长度小于 k，不可能构造出长度为 k 的 temp
        if (temp.size() + (n - cur + 1) < k) {
            return;
        }
        // 记录合法的答案
        if (temp.size() == k) {
            ans.add(new ArrayList<Integer>(temp));
            return;
        }
        // 考虑选择当前位置
        temp.add(cur);
        dfs(cur + 1, n, k);
        temp.remove(temp.size() - 1);
        // 考虑不选择当前位置
        dfs(cur + 1, n, k);
    }
}
```

`Backtracking `

```java
class Solution {
     public static List<List<Integer>> combine(int n, int k) {
		List<List<Integer>> combs = new ArrayList<List<Integer>>();
		combine(combs, new ArrayList<Integer>(), 1, n, k);
		return combs;
	}
	public static void combine(List<List<Integer>> combs, List<Integer> comb, int start, int n, int k) {
		if(k==0) {
			combs.add(new ArrayList<Integer>(comb));
			return;
		}
		for(int i=start;i<=n;i++) {
			comb.add(i);
			combine(combs, comb, i+1, n, k-1);
			comb.remove(comb.size()-1);
		}
	}
}
```

## Pow(x,n)

`divide and conquer log(n)`    

 **pow(x, n) ->sub problem : pow(x,n/2)**

```java
public double myPow(double x, int n) {
        if (n == 0) return 1.0;
        if (n == Integer.MIN_VALUE) {
            x = x * x;
            n = n / 2;
        }
        if (n < 0) {
            n = -n;
            x = 1 / x;
        }
        return n % 2 == 0 ? myPow(x * x , n / 2) : x * myPow(x * x, n / 2);
    }
```

## subset

`backtracking`

```java
class Solution {
    
    List<Integer> list = new ArrayList<>();
    List<List<Integer>> result = new ArrayList<List<Integer>>();
    
    public List<List<Integer>> subsets(int[] nums) {
        dfs(0,nums);
        return result;
    }
    
    public void dfs(int idx, int[] nums) {
        if (idx == nums.length) {
            result.add(new ArrayList<>(list));
            return;
        }
        dfs(idx+1,nums);
        list.add(nums[idx]);
        dfs(idx+1,nums);
        list.remove(list.size() - 1);
    }
    
}
```

`backtracking II`

```java
class Solution{
    List<List<Integer>> list = new ArrayList<>();
    List<Integer> tempList = new ArrayList<>();

    public List<List<Integer>> subsets(int[] nums) {
        Arrays.sort(nums);
        backtrack(0, nums);
        return list;
    }
    private void backtrack(int start, int[] nums) {
        list.add(new ArrayList<>(tempList));
        for (int i = start; i < nums.length; i++) {
            tempList.add(nums[i]);
            backtrack(i + 1,nums);
            tempList.remove(tempList.size() - 1);
        }
    } 
}
```

## subset II

`backtracking`

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> list = new ArrayList<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        backtracking(0,nums);
        return res;
    }
    public void backtracking(int pos,int[] nums) {
        res.add(new ArrayList<>(list));
        for (int i = pos; i < nums.length; i ++) {
            if (i > pos && nums[i] == nums[i - 1]) continue;
            list.add(nums[i]);
            backtracking(i + 1, nums);
            list.remove(list.size() - 1);
        }
    }
}
```

## permutations

`backtracking`

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        helper(result, new ArrayList<>(), nums);
        return result;
    }
    public void helper(List<List<Integer>> result, List<Integer> list,int[] nums) {
        if (nums.length == list.size()) {
            result.add(new ArrayList<>(list));
        } else {
            for (int i = 0; i < nums.length; i ++) {
                if (list.contains(nums[i])) continue;
                list.add(nums[i]);
                helper(result, list, nums);
                list.remove(list.size() - 1);
            }
        }
    }
}
```

## permutations II

`backtracking`

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    List<Integer> list = new ArrayList<>();
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        helper(nums, new boolean[nums.length]);
        return result;
    }
    public void helper(int[] nums, boolean[] used) {
        if (list.size() == nums.length) {
            result.add(new ArrayList<>(list));
        } else {
            
            for (int i = 0; i < nums.length; i++) {
                if (used[i] || i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;
                used[i] = true;
                list.add(nums[i]);
                helper(nums,used);
                used[i] = false;
                list.remove(list.size() - 1);
            }
        }
    }
}
```

## Combination Sum

`backtracking`

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    List<Integer> list = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        helper(candidates,target,0);
        return result;
    }
    public void helper(int[] nums, int remain, int start) {
        if (remain < 0) return;
        else if (remain == 0) result.add(new ArrayList<>(list));
        else {
            for (int i = start; i < nums.length; i++) {
                list.add(nums[i]);
                helper(nums, remain - nums[i], i); //not i + 1 cause we need to use the same element
                list.remove(list.size() - 1);
            }
        }
    }
}
```



## Combination Sum II

`backtracking`

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    List<Integer> list = new ArrayList<>();
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        helper(candidates, target,0);
        return result;
    }
    public void helper(int[] nums, int remain, int start) {
        if (remain < 0) return;
        else if (remain == 0) result.add(new ArrayList<>(list));
        else {
            for (int i = start; i < nums.length; i ++) {
                if (i > start && nums[i] == nums[i - 1]) continue; //skip duplicates
                list.add(nums[i]);
                helper(nums, remain - nums[i],i + 1); //i + 1 -> not use the same element
                list.remove(list.size() - 1);
            }
        }
    }
}
```

## Palindrome Partitioning

`backtracking`

```java
class Solution {
    List<List<String>> result = new ArrayList<>();
    List<String> list = new ArrayList<>();

    public List<List<String>> partition(String s) {
        helper(s,0);
        return result;
    }

    public void helper(String s, int start) {
        if (s.length() == start) {
            result.add(new ArrayList<>(list));
        } else {
            for (int i = start; i < s.length(); i++) {
                if (isPalindrome(s,start,i)) {
                    list.add(s.substring(start, i + 1)); //split s between start and i 
                    helper(s,i + 1);
                    list.remove(list.size() -  1);
                }
            }
        }
    }

    //judge a string is palindrome method
    public boolean isPalindrome(String s, int low, int high) {
        while(low < high) {
            if (s.charAt(low++) != s.charAt(high--)) {
                return false;
            }
        }
        return true;
    }
}
```

## majority element

`O(1) time O(n) space` 

摩尔投票法

核心理念为 **票数正负抵消**

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

## binary tree level order traversal

`DFS recursion `

```java
class Solution {
    List<List<Integer> res = new ArrayList<>();
    public List<List<Integer>> levelOrder(TreeNode root) {
		helper(root, 0);
        return res;
    }
    public void helper(TreeNode node, int height) {
        if (node == null) return;
        if (height == res.size()) res.add(new ArrayList<Integer>());
        res.get(height).add(node.val);
        helper(node.left, height + 1));
        helper(node.right, height + 1);
    }
}
```

`BFS deque`

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);
        while(!queue.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            int cnt = queue.size();
            for (int i = 0; i < cnt; i++) {
                TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            res.add(list);
        }
        return res;
    }
}
```

## binary tree level order traversal ii

利用Api 每一层进行插入的时候都设置索引为0 即头插法 因为返回的是一个接口 所以实现类可以使用LinedList O(1) 复杂度插入首部

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new LinedList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);
        while(!queue.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            int cnt = queue.size();
            for (int i = 0; i < cnt; i++) {
                TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            res.add(0, list);
        }
        return res;
    }
}
```



## find the largest value o each tree row

`DFS`

E set(int index, E element) Replace the element at the specified position in this list with the specified element(optional operation)

```java
class Solution {
   	List<Integer> res = new ArrayList<>();
    public List<Integer> largestValues(TreeNode root) {
        helper(root, 0);
        return res;
    }
    public void helper(TreeNode node, int level) {
        if (node == null) return;
        if (level == res.size()) {
            res.add(node.val);
        } else {
            res.set(level, Math.max(res.get(level), node.val));
        }
        helper(node.left, level + 1);
        helper(node.right, level + 1);
    }
}
```

`BFS` 维护queue   注意空树的情况

```java
class Solution {
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        if (root == null) return res;
        queue.offer(root);
        while (!queue.isEmpty()) {
            int max = Integer.MIN_VALUE;
            for (int i = 0; i < queue.size(); i++) {
                TreeNode node = queue.poll();
                max = Math.max(max, node.val);
                if (node.left != null) queue.offer(node.left);
                if (node,right != null) queue.offer(node.right);
            }
            res.add(max);
        }
        return res;
    }
}
```

## word ladder

`two end BFS` faster than one end

**已知目标顶点的情况下，可以分别从起点和目标顶点（终点）执行广度优先遍历，直到遍历的部分有交集。这种方式搜索的单词数量会更小一些；更合理的做法是，每次从单词数量小的集合开始扩散** 

```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
	    Set<String> beginSet = new HashSet<>(), endSet = new HashSet<>();
        Set<String> dict = new HashSet<>(wordList);
        if (!dict.contains(endWord)) return 0;
        beginSet.add(beginWord);
        endSet.add(endWord);
        Set<String> visited = new HashSet<>();
        int len = 1;
        while (!beginSet.isEmpty() && !endSet.isEmpty()) {
            if (beginSet.size() > endSet.size()) {
                Set<String> set = beginSet;
                beginSet = endSet;
                endSet = set;
            }
            Set<String> temp = new HashSet<>();
            for (String word :beginSet){
                char[] chs = word.toCharArray();
                for (int i = 0; i < chs.length; i++) {
                    for (char ch = 'a'; ch <= 'z'; ch++) {
                    	 char old = chs[i];
                         chs[i] = ch;
                         String target = String.valueOf(chs);
                         if (endSet.contains(target)) return ++len;
                         if (!visited.contains(target) && dict.contains(target)) {
                             temp.add(target);
                             visited.add(target);
                         }
                        chs[i] = old;
                    }
                }
            }
            beginSet = temp;
            len ++;
        }
        return 0;
    }
}
```

## flood fill

`简化版本的网格dfs问题`区别于岛屿问题的关键是 `不需要判断当前节点是否遍历过`

因为通过dfs修改颜色之前要保存之前的颜色 而到下一次相邻的网格修改之后 再会来判断 颜色已经改变了 所以不会造成死循环

但是需要注意的是如果一开始的网格颜色和涂鸦颜色相同 会造成死循环 所以一开始做一个判断 如果相同 直接返回原数组

`dfs`

```java
Class Solution {
    int newColor;
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        this.newColor = newColor;
        if (image[sr][sc] != newColor) {
            bfs(image, sr, sc, image[sr][sc]);
        }
        return image;
    }

    private void bfs(int[][] image, int i, int j, int color) {
        
        if (!valid(image, i, j)) return;
        if (image[i][j] != color) return;
        
        int prev = image[i][j];
        image[i][j] = newColor;
        
        bfs(image, i - 1, j, prev);
        bfs(image, i + 1, j, prev);
        bfs(image, i, j - 1, prev);
        bfs(image,i, j + 1, prev);

    }

    private boolean valid(int[][] image, int i, int j) {
        return (i >= 0 && i < image.length) && (j >= 0 && j < image[0].length);
    }
}
```

## rottiong orange

+ 为什么要用BFS

bfs 适合求最短路径问题 而且通过循环控制可以做到层序遍历

```java
class Solution {
    public int orangesRotting(int[][] grid) {
        Deque<int[]> queue = new LinkedList<>();
        //表示新鲜橘子的个数
        int cnt = 0;
        //统计新鲜橘子的个数以及将所有烂橘子放入队列等待遍历
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) cnt++;
                else if (grid[i][j] == 2) queue.add(new int[]{i, j});
            }
        }
        //记录层数
        int round = 0;
        while (cnt > 0 && !queue.isEmpty()) {
            //获取当前层烂橘子个数
            int n = queue.size();
            //记录层数
            round++;
            //依次进行遍历处理 标记每个烂橘子四个方向 并统计结果
            for (int i = 0; i < n; i++) {
                int[] curr = queue.poll();
                int x = curr[0];
                int y = curr[1];
                if (x - 1 >= 0 && grid[x - 1][y] == 1) {
                    cnt--;
                    grid[x - 1][y] = 2;
                    queue.add(new int[]{x - 1, y});
                }
                if (x + 1 < grid.length && grid[x + 1][y] == 1) {
                    cnt--;
                    grid[x + 1][y] = 2;
                    queue.add(new int[]{x + 1, y});
                }
                if (y - 1 >= 0 && grid[x][y - 1] == 1) {
                    cnt--;
                    grid[x][y - 1] = 2;
                    queue.add(new int[]{x, y - 1});
                }
                if (y + 1 < grid[0].length && grid[x][y + 1] == 1) {
                    cnt--;
                    grid[x][y + 1] = 2;
                    queue.add(new int[]{x, y + 1});
                }
            }
        }
        return cnt > 0 ? -1 : round;
    }
}
```

`optimization` 遍历四个方向可以优化 通过数组存储四个方向的横纵坐标差值 然后遍历4个方向

如下方法实现 四个方向的移动

```python
# 设初始点为 (i, j)
for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 上、下、左、右
    i + di, j + dj
```

```java
int[][] dir = {{-1,0},{1,0},{0,-1},{0,1}};//四个方向移动

for (int j = 0; j < 4; j++) {
     int x = curr[0] + dir[j][0];
     int y = curr[1] + dir[j][1];
     if(x >= 0 && x < grid.length && y >= 0 && y < grid[0].length
                    && grid[x][y] == 1) {
          cnt--;
          grid[x][y] = 2;
          queue.add(new int[]{x, y});
    }
}
```

## 01 matrix

`bfs`

```java
class Solution {
    public int[][] updateMatrix(int[][] matrix) {
        // 首先将所有的 0 都入队，并且将 1 的位置设置成 -1，表示该位置是 未被访问过的 1
        Queue<int[]> queue = new LinkedList<>();
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    queue.offer(new int[] {i, j});
                } else {
                    matrix[i][j] = -1;
                } 
            }
        }
        
        int[] dx = new int[] {-1, 1, 0, 0};
        int[] dy = new int[] {0, 0, -1, 1};
        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            int x = point[0], y = point[1];
            for (int i = 0; i < 4; i++) {
                int newX = x + dx[i];
                int newY = y + dy[i];
                // 如果四邻域的点是 -1，表示这个点是未被访问过的 1
                // 所以这个点到 0 的距离就可以更新成 matrix[x][y] + 1。
                if (newX >= 0 && newX < m && newY >= 0 && newY < n 
                        && matrix[newX][newY] == -1) {
                    matrix[newX][newY] = matrix[x][y] + 1;
                    queue.offer(new int[] {newX, newY});
                }
            }
        }

        return matrix;
    }
}
```

`dp`

![image-20211228104903049](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20211228104903049.png)

```java
class Solution {
  public int[][] updateMatrix(int[][] matrix) {
    int m = matrix.length, n = matrix[0].length;
    int[][] dp = new int[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        dp[i][j] = matrix[i][j] == 0 ? 0 : 10000;
      }
    }

    // 从左上角开始
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (i - 1 >= 0) {
          dp[i][j] = Math.min(dp[i][j], dp[i - 1][j] + 1);
        }
        if (j - 1 >= 0) {
          dp[i][j] = Math.min(dp[i][j], dp[i][j - 1] + 1);
        }
      }
    }
    // 从右下角开始
    for (int i = m - 1; i >= 0; i--) {
      for (int j = n - 1; j >= 0; j--) {
        if (i + 1 < m) {
          dp[i][j] = Math.min(dp[i][j], dp[i + 1][j] + 1);
        }
        if (j + 1 < n) {
          dp[i][j] = Math.min(dp[i][j], dp[i][j + 1] + 1);
        }
      }
    }
    return dp;
  }
}
```



## max area of island

`岛屿问题-->网格DFS问题`

- 如何遍历？

`网格DFS遍历的代码框架`

```java
void dfs(int[][] grid, int r, int c) {
    // 判断 base case
    // 如果坐标 (r, c) 超出了网格范围，直接返回
    if (!inArea(grid, r, c)) {
        return;
    }
    // 访问上、下、左、右四个相邻结点
    dfs(grid, r - 1, c);
    dfs(grid, r + 1, c);
    dfs(grid, r, c - 1);
    dfs(grid, r, c + 1);1
}

// 判断坐标 (r, c) 是否在网格中
boolean inArea(int[][] grid, int r, int c) {
    return 0 <= r && r < grid.length 
        	&& 0 <= c && c < grid[0].length;
}
```

- 网络结构是一个图 如何避免重复遍历同一个节点？

标记已经遍历过的节点

```java
void dfs(int[][] grid, int r, int c) {
    // 判断 base case
    if (!inArea(grid, r, c)) {
        return;
    }
    // 如果这个格子不是岛屿，直接返回
    if (grid[r][c] != 1) {
        return;
    }
    grid[r][c] = 2; // 将格子标记为「已遍历过」
    
    // 访问上、下、左、右四个相邻结点
    dfs(grid, r - 1, c);
    dfs(grid, r + 1, c);
    dfs(grid, r, c - 1);
    dfs(grid, r, c + 1);
}

// 判断坐标 (r, c) 是否在网格中
boolean inArea(int[][] grid, int r, int c) {
    return 0 <= r && r < grid.length 
        	&& 0 <= c && c < grid[0].length;
}
```

`DFS solution`

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int res = 0;
        for (int i = 0;i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    int area = dfs(i, j, grid);
                    res = Math.max(res, area);
                }
            }
        }
        return res;
    }
    public int dfs(int r, int c, int[][] grid) {
        if (!isArea(r, c, grid)) return 0;
        if (grid[r][c] != 1) return 0;
        grid[r][c] = 2;
        return 1
        + dfs(r + 1, c, grid)
        + dfs(r - 1, c, grid)
        + dfs(r, c + 1, grid)
        + dfs(r, c - 1, grid);
    }
    public boolean isArea(int r, int c, int[][]grid) {
        return (0 <= r && r < grid.length && 0 <= c && c < grid[0].length);
    }
}
```

## making a large island

## 	island perimeter

`math for loop`

```java
class Solution {
    public int islandPerimeter(int[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    res += 4;
                    //如果当前格子的上方也是陆地 则交界处 周长 - 2
                    if (i > 0 && grid[i -1][j] == 1) {
                        res -= 2;
                    }
                    //如果当前格子的左方也是陆地 则交界处 周长 - 2
                    if (j > 0 && grid[i][j - 1] == 1) {
                        res -= 2;
                    }
                }
            }
        }
        return res;
    }
}
```

`DFS`

```java
class Solution {
    int peimeter = 0;
    public int islandPerimeter(int[][] grid) {
        for (int i = 0; i < grid.length; i++) {
            for(int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) dfs(i, j, grid);
            }
        }
        return peimeter;
    }
    public void dfs(int r, int c, int[][]grid) {
        //地图的边界
        if (!isArea(r, c, grid)) {
            peimeter ++;
            return;
        }
        //海洋的边界
        if (grid[r][c] == 0) {
            peimeter ++;
            return;
        }
        //已经遍历过的格子
        if (grid[r][c] == 2) return;
        grid[r][c] = 2;
        dfs(r + 1, c, grid);
        dfs(r - 1, c, grid);
        dfs(r, c + 1, grid);
        dfs(r, c - 1, grid);
    }
    public boolean isArea(int r, int c, int[][]grid) {
        return 0 <= r && r < grid.length && 0 <= c && c < grid[0].length;
    }
}
```

## number of island

`DFS`

```java
class Solution {
    public int numIslands(char[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    res ++;
                    dfs(i, j, grid);
                }
                
            }
        }
        return res;
    }
    public void dfs(int r, int c, char[][] grid) {
        if (!isArea(r, c, grid)) return;
        if (grid[r][c] != '1') return;
        grid[r][c] = '2';
        dfs(r + 1, c, grid);
        dfs(r - 1, c, grid);
        dfs(r, c + 1, grid);
        dfs(r, c - 1, grid);
    }
    public boolean isArea(int r, int c, char[][] grid){
        return 0 <= r && r < grid.length && 0 <= c && c < grid[0].length;
    }
}
```

## mine sweeper

`DFS`

1. If click on a mine ('`M`'), mark it as '`X`', stop further search.
2. If click on an empty cell ('`E`'), depends on how many surrounding mine:
   2.1 Has surrounding mine(s), mark it with number of surrounding mine(s), stop further search.
   2.2 No surrounding mine, mark it as '`B`', continue search its `8` neighbors.

```java
class Solution {
    public char[][] updateBoard(char[][] board, int[] click) {
        int m = board.length, n = board[0].length;
        int row = click[0], col = click[1];
        if (board[row][col] == 'M') {
            board[row][col] = 'X';
        } else {
            int count = 0;
            for (int i = -1; i < 2; i ++) {
                for(int j = -1; j < 2; j ++) {
                    //跳过click的位置
                    if (i == 0 && j == 0) continue;
                    //考虑边界问题
                    int r = row + i, c = col + j;
                    if (r < 0 || r > m - 1  || c < 0 || c > n - 1) continue;
                    if (board[r][c] == 'M' || board[r][c] == 'X') count ++; 
                }
            }
            if (count > 0){
                 board[row][col] = (char)(count += '0');
            }else {
                board[row][col] = 'B';
                for (int i = -1; i < 2; i ++) {
                    for(int j = -1; j < 2; j ++) {
                        //跳过click的位置
                        if (i == 0 && j == 0) continue;
                        //考虑边界问题
                        int r = row + i, c = col + j;
                        if (r < 0 || r > m - 1  || c < 0 || c > n - 1) continue;
                        if (board[r][c] == 'E') updateBoard(board, new int[]{r,c}); 
                    }
                }
            }
        }
        return board;
    }
}
```

## assign cookies

`贪心算法`

```java
class Solution {
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int i = 0, j = 0, res = 0;
         while (i < g.length && j < s.length) {
             if (g[i] <= s[j]) {
                 res++;
                 i++;
                 j++;
             } else {
                 j++;
             }
         }
        return res;
    }
}
```

## jump game

`遍历每个格子 更新能够到达的最远距离 如果 最远距离始终大于格子的下标 返回true`

```java
class Solution {
    public boolean canJump(int[] nums) {
        int k = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > k) return false;
            k = Math.max(k, i + nums[i]);
        }
        return true;
    }
}
```

## jump game II

```java
public int jump(int[] nums) {
    int end = 0;
    int maxPosition = 0; 
    int steps = 0;
    for(int i = 0; i < nums.length - 1; i++){
        //找能跳的最远的
        maxPosition = Math.max(maxPosition, nums[i] + i); 
        if( i == end){ //遇到边界，就更新边界，并且步数加一
            end = maxPosition;
            steps++;
        }
    }
    return steps;
}
```



## walking robor simulation

`巧妙之处在于方向的表示以及处理遇到障碍物的情况`

```java
class Solution {
    public int robotSim(int[] commands, int[][] obstacles) {
        int[][] diractions = {{0,1},{1,0},{0,-1},{-1,0}};
        int x = 0, y = 0, diraction = 0;
        int res = 0;
        Set<String> obstaclesSet = new HashSet<>();
        for (int[] ob : obstacles) {
            obstaclesSet.add(ob[0] + " " + ob[1]);
        }
        for (int i = 0; i < commands.length; i++) {
            if (commands[i] == -1) {
                diraction = (diraction + 1) % 4;
            } else if (commands[i] == -2) {
                diraction = (diraction + 3) % 4;
            } else {
                int step = 0;
                while (step < commands[i] && !obstaclesSet.contains(
                    (x + diractions[diraction][0]) + " " + (y + diractions[diraction][1]))) {
                    step ++;
                    x += diractions[diraction][0];
                    y += diractions[diraction][1];
                }
            }
            res = Math.max(res, x * x + y * y);
        }
        return res;
    }
}
```

## sqrt x

`二分法`  细节较多 基于模板要注意细节 这里返回的right 而不是mid 同时l初始化为1 而不是0

```java
public int mySqrt(int x) {
        int left = 1, right = x;
        int mid = 0;
        while (left <= right) {
            mid =  left + (right - left) / 2;
            if (mid  == x / mid) {
                return mid;
            } else if (mid  < x / mid) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return right;
    }
```

```java
public int mySqrt(int x) {
    long res = x;
    while (res * res > x) {
        res  = (res + x / res) / 2;
    }
    return (int)res;
}
```

## non decreasing array

1. 判断第一个元素

2. 判断最后一个元素

3. 判断中间元素和前一个、后一个元素的大小关系

   3.1 判断前一个和后一个元素的大小关系 <

```java
class Solution {
    public boolean checkPossibility(int[] nums) {
        int count = 0;
        if (nums.length == 1) return true;
        if (nums[0] > nums[1]) {
            nums[0] = nums[1];
            count ++;
        }
        if (nums[nums.length - 1] < nums[nums.length - 2]) {
            nums[nums.length - 1] = nums[nums.length - 2];
            count ++;
        }
        for(int i = 1; i < nums.length - 1; i++) {
            if (nums[i] > nums[i + 1] || nums[i] < nums[i - 1]) {
               if(nums[i + 1] > nums[i - 1]) {
                    nums[i] = nums[i - 1];
                    count ++;
               } else if (nums[i + 1] < nums[i - 1]) {
                    nums[i + 1] = nums[i];
                    count ++;
               }
            }
        }
        return count < 2;
    } 
}
```

## search in rotated sorted array

```java
public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int start = 0;
        int end = nums.length - 1;
        int mid;
        while (start <= end) {
            mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            //前半部分有序,注意此处用小于等于
            if (nums[start] <= nums[mid]) {
                //target在前半部分
                if (target >= nums[start] && target < nums[mid]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
            } else {
                if (target <= nums[end] && target > nums[mid]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }

        }
        return -1;
    }
```

## find minimum in rotated sorted array

`二分法`

```java
class Solution {
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int mid = l + (r - l >> 1);
            if (nums[mid] > nums[r]) l = mid + 1;
            else if (nums[mid] < nums[r]) r = mid;
            else r--;
        }
        return nums[l];
    }
}
```

## find minimum in rotated sorted array ii

​																								`二分法 解法同上`

## search a 2d matrix

`二分法`

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int left = 0, right = m * n - 1;
        int mid, idxEle;
        while (left <= right) {
            //加减运算优先于位运算
            mid = left + (right - left >> 1);
            //二位矩阵的坐标 与 索引 mid 的转换
            idxEle = matrix[mid / n][mid % n];
            if (idxEle == target) return true;
            else if (idxEle > target) {
                right = mid - 1;
            } else left = mid + 1;
        }
        return false;
    }
}
```

## search a 2d matrix ii

从左下角开始查找 具备以下性值 右边元素均大于matrix[m] [n] 上边元素均小于matrix[m] [n]

iteration terminator :  m >=0 && n < matrix[0].length

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) return false;
        //m: row n: colomn
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



## unique paths

`DP`

```java
class Solution {
    public int uniquePaths(int m, int n) {
		int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; i < n; j++) {
            dp[0][j] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }
}
```

`same time but save space`

```java
class Solution {
    public int uniquePaths(int m, int n) {
		int[] curr = new int[n];
        Arrays.fill(curr, 1);
        for (int i = 1; i < m; i++) {
            for (intj = 1; j < n; j++) {
                curr[j] += curr[j - 1];
            }
        }
        return curr[n - 1];
    }
}
```

## unique paths II

`DP`

```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }
        
        // 定义 dp 数组并初始化第 1 行和第 1 列。
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m && obstacleGrid[i][0] == 0; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n && obstacleGrid[0][j] == 0; j++) {
            dp[0][j] = 1;
        }

        // 根据状态转移方程 dp[i][j] = dp[i - 1][j] + dp[i][j - 1] 进行递推。
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 0) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return p[m - 1][n - 1];
    }
}
```

## interleaving string

`dp`

![image-20211214110355130](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20211214110355130.png)

```java
class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {
        int m = s1.length(), n = s2.length();
        if (s3.length() != m + n) return false;
        // 动态规划，dp[i,j]表示s1前i字符能与s2前j字符组成s3前i+j个字符；
        boolean[][] dp = new boolean[m+1][n+1];
        dp[0][0] = true;
        for (int i = 1; i <= m && s1.charAt(i-1) == s3.charAt(i-1); i++) dp[i][0] = true; // 不相符直接终止
        for (int j = 1; j <= n && s2.charAt(j-1) == s3.charAt(j-1); j++) dp[0][j] = true; // 不相符直接终止
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (dp[i - 1][j] && s3.charAt(i + j - 1) == s1.charAt(i - 1))
                    || (dp[i][j - 1] && s3.charAt(i + j - 1) == s2.charAt(j - 1));
            }
        }
        return dp[m][n];
    }
}
```



## longest common subsequence

`DP`  转化为二维数组的递推

![image-20210216171453479](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210216171453479.png)

```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
		int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        for (int i = 0; i < text1.length; i++) {
            for (int j = 0; j < text2.length(); j++) {
                if (text1.charAt(i) == text2.charAt(j)) {
                    dp[i + 1][j + 1] = 1 + dp[i][j];
                } else {
                    dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
                }
            }
        }
        return dp[text1.length()][text2.length()];
    }
}
```

## longest common substring

`DP` 转化为二维数组 与最长公共子序列类似

```java
class Solution {
	public int longestCommonSubstring(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
                if (text1.charAt(i) == text2.charAt(j) {
					dp[i + 1][j + 1] = dp[i][j] + 1;
                } else {
                    dp[i + 1][j + 1] = 0;
                }
            }
        }
        return dp[m][n];
    }
}
```

## longest substring without repeating characters 

`滑动窗口`

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int[] index = new int[128];
        int ans = 0;
        for(int j=0,i=0;j<s.length();j++){
            i=Math.max(index[s.charAt(j)],i);
            ans = Math.max(ans,j-i+1);
            index[s.charAt(j)] = j+1;
        }
        return ans;
    
    }
}" 

```

`dp`

+ 状态定义： 设动态规划列表 dpdp ，dp[j]dp[j] 代表以字符 s[j]s[j] 为结尾的 “最长不重复子字符串” 的长度。
+ 转移方程： 固定右边界 jj ，设字符 s[j]s[j] 左边距离最近的相同字符为 s[i]s[i] ，即 s[i] = s[j]s[i]=s[j] 。

1. 当 i < 0i<0 ，即 s[j]s[j] 左边无相同字符，则 dp[j] = dp[j-1] + 1dp[j]=dp[j−1]+1 ；
2. 当 dp[j - 1] < j - idp[j−1]<j−i ，说明字符 s[i]s[i] 在子字符串 dp[j-1]dp[j−1] 区间之外 ，则 dp[j] = dp[j - 1] + 1dp[j]=dp[j−1]+1 ；
3. 当 dp[j - 1] \geq j - idp[j−1]≥j−i ，说明字符 s[i]s[i] 在子字符串 dp[j-1]dp[j−1] 区间之中 ，则 dp[j]dp[j] 的左边界由 s[i]s[i] 决定，即 dp[j] = j - idp[j]=j−i ；

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int temp;
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            int left = map.getOrDefault(s.charAt(i), i);
            temp = temp < i - left ? temp + 1 : i - left;
            res = Math.max(res, temp);
        }
        return res;
    }
}
```





## triangle

'Bottom-up' DP, on the other hand, is very straightforward: we start from the nodes on the bottom row; the min pathsums for these nodes are the values of the nodes themselves. From there, the min pathsum at the ith node on the kth row would be the lesser of the pathsums of its two children plus the value of itself, i.e.:

```
minpath[k][i] = min( minpath[k+1][i], minpath[k+1][i+1]) + triangle[k][i];
```

Or even better, since the row minpath[k+1] would be useless after minpath[k] is computed, we can simply set minpath as a 1D array, and iteratively update itself:

```
For the kth level:
minpath[i] = min( minpath[i], minpath[i+1]) + triangle[k][i];
```

```java
public int minimumTotal(List<List<Integer>> triangle) {
        int row = triangle.size();
        int[] dp = new int[row];
        //init 1D array dp
        for (int i = 0; i < triangle.get(row - 1).size(); i++) {
            dp[i] = triangle.get(row - 1).get(i);
        }
        //bottom-up 
        for (int i = row - 2; i >= 0; i--) {//for last second layer
            for (int j = 0; j < i + 1; j ++) {//for every element 
                dp[j] = Math.min(dp[j],dp[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];
    }
```

## pascals triangle ii

题目：根据行索引 求行 （from idx 0）

**同一行的相邻组合数的关系**

![image-20211109133159877](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20211109133159877.png)

第row行 有 row + 1 列

第0列 也就是list.add(1) 从第一列开始 遍历到 第row列

每一列都与上一列的值有关系 也就是

currVal = prevVal * (rowNum - colNum + 1) / colNum

即 row.add((int)((long) list.get(i - 1) * (rowNum - i + 1) / i)

```java
class Solution {
    public List<Integer> getRow(int rowIndex) {
        List<Integer> list  = new ArrayList<>();
        list.add(1);
        for (int i = 1; i <= rowIndex; i++) {
            long temp = (long)list.get(i - 1) * (rowIndex - i + 1) / i;
            list.add((int)temp);
        }
        return list;
    }
}
```

## maximum subarray

`dp`

```java
class Solution{
    public int maxSubArray(int[] nums) {
        int res = Integer.MIN_VALUE, sum = 0;
        for (int num : nums) {
            if (sum > 0) {
                sum += num;
            } else {
                sum = num;
            }
            res = Math.max(res, sum);
        }
        return res;
    }
}
```

## maximum product subarray

`dp`

遍历数组的时候计算当前最大值， 考虑负数的情况如果当前num < 0 则 imax 和 imin 交换

```java
class Solution{
    public int maxProduct(int[] nums) {
        int max = Integer.MIN_VALUE, imax = 1, imin = 1;
        for (int num : nums) {
            if (num < 0) {
                int tmp = imax;
                imax = imin;
                imin = tmp;
            }
            imax = Math.max(imax * num, num);
            imin = Math.min(imin * num, num);
            max = Math.max(max, imax);
        }
        return max;
    }
}
```

## coin change

`dp`

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        for(int i = 1; i <= amount; i++) {
            int min = amount + 1;
            for(int coin : coins) {
                if(i >= coin) min = Math.min(dp[i - coin] + 1, min);
            }
            dp[i] = min;
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
}
```

## partiion equal subset sum

`dp` `0-1 bag`

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int m = nums.length, sum = 0;
        for (int num : nums) sum += num;
        if ((sum & 1) == 1) return false;
        int n = sum >> 1;
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = j >= nums[i - 1] ? dp[i - 1][j] || dp[i - 1][j - nums[i - 1]] : dp[i - 1][j];
            }
        }
        return dp[m][n];
    }
}
```

`1d dp`

1. 降维：因为`dp[i][j]` 取决于`dp[i - 1][j - nums[i - 1]] || dp[i - 1][j]` 即第一维当前都是取决于上一层的关系 所以可以使用滚动数组覆盖的方式只保留上一层的状态（与上一层之前的状态无关）
2. 剪枝: 当`j < nums[i - 1]`时 说明当前层的背包已经放不下新的数字了（不满足 sum/2) 所以此时的`dp[j]`只取决于上一层此时的状态 也就是当前层我不取数字了 至于我上一层状态有关 所以可以剪枝
3. 状态问题的考虑：另外需要考虑的是如果采用正向遍历的方式 从`nums[j - 1] -> n`的话 因为`j - nums[i]`是严格小于`j`的 所以`dp[j - nums[i - 1]]`会计算出来  而此时的`dp[j - nums[i - 1]]`是当前层的状态也就是新状态 而我们需要计算的`dp[j] = dp[j] || dp[j - num[i - 1]`这两个值是上一层的状态 所以出现了状态不一致的情况 解决思路便是通过倒序的方式遍历 这样就可以保证状态的顺序 而且倒序也是剪枝的前提条件

综合来看 想要理解尽量还是结合画图从二维开始推演 从而理解dp的逻辑

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) sum += num;
        if ((sum & 1) == 1) return false;
        int n = sum >> 1;
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        for (int i = 1; i <= nums.length; i++) {
            for (int j = n; j >= nums[i - 1]; j--) {
                dp[j] |= dp[j - nums[i - 1]];
            }
        }
        return dp[n];
    }
}
```

## house robber

`2D dp`

开辟一个维度来保存偷或者不偷的状态

dp方程

```java
a[i][0] = Max(a[i-1][0],a[i-1][1])
a[i][1] = a[i-1][0] + nums[i]
```

```java
class Solution {
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int[][] dp = new int[nums.length][2];
        dp[0][0] = 0;
        dp[0][1] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1]= dp[i - 1][0] + nums[i];
        }
        return Math.max(dp[nums.length - 1][0], dp[nums.length - 1][1]);
    }
}
```

`1D dp`

a[i] : 0 ... i 天 的最大值 状态转移方程即

```java
a[i] = max(a[i-1], a[i - 2] + nums[i])
```

```java
class Solution {
    public int rob(int[] nums) {
       int n = nums.length;
        if (n == 1) return nums[0];
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(dp[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[n - 1];
    }
}
```

`optimal dp`

上一个的状态转移方程其实类似于Fibonacci数列 所以只需要保存三个变量即可

```java
class Solution{
    public int rob(int[] nums) {
		int prevMax = 0;
        int currMax = 0;
        for (int x : nums) {
			int temp = currMax;
            currMax = Math.max(prevMax + x, currMax);
            prevMax = temp;
        }
        return currMax;
    }
}
```

## house robber II

***Java.util.Arrays.copyOfRange() Method***

**Description**

The java.util.Arrays.copyOfRange(short[] original, int from, int to) method copies the specified range of the specified array into a new array.The final index of the range (to), which must be greater than or equal to from, may be greater than original.length, in which case (short)0 is placed in all elements of the copy whose index is greater than or equal to original.length - from. The length of the returned array will be to - from.

**Declaration**

```
public static short[] copyOfRange(short[] original, int from, int to)
```

**Parameters**

- original − This is the array from which a range is to to be copied.
- from − This is the initial index of the range to be copied, inclusive.
- to − This is the final index of the range to be copied, exclusive.

`optimal dp` 

**核心思路**  将环形队列拆分为两个队列 返回两个结果的较大值 

```java
class Solution {
	public int rob(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        return Math.max(myRob(Arrays.copyOfRange(nums, 0, nums.length - 1)), myRob(Arrays.copyOfRange(nums, 1,nums.length)));
    }
    public int myRob(int[] nums) {
		int prevMax = 0, currMax = 0;
        for (int num : nums) {
			int temp = currMax;
            currMax = Math.max(prevMax + num, currMax);
            prevMax = temp;
        }
        return currMax;
    }
}
```

## house rob III

`暴力递归`

核心思想 每次递归计算`爷爷节点 + 4个孙子节点` 和 `两个儿子节点`的较大值 

会超时 问题是 每次下探爷爷节点 都要重新计算儿子节点的值 产生大量重复计算

这也是dp的通常优化点

```java
class Solution {
    public int rob(TreeNode root) {
        if (root == null) return 0;
        int total = root.val;
        if(root.left != null) {
            total += rob(root.left.left) + rob(root.left.right);
        } 
        if (root.right != null) {
            total += rob(root.right.right) + rob(root.right.left);
        }
        return Math.max(total, rob(root.left) + rob(root.right));
            
    }
}
```

`递归 + 记忆化搜索` O（N）

通常使用数组存储中间结果 对于二叉树可以使用**hash**表来存

TreeNode 当做 key，能偷的钱当做 value

```java
class Solution {
    Map<TreeNode, Integer> map = new HashMap<>();
    public int rob(TreeNode root) {
        if (root == null) return 0;
        if (map.containsKey(root)) return map.get(root);
        int total = root.val;
        if(root.left != null) {
            total += rob(root.left.left) + rob(root.left.right);
        } 
        if (root.right != null) {
            total += rob(root.right.right) + rob(root.right.left);
        }
        int res = Math.max(total, rob(root.left) + rob(root.right));
        map.put(root, res);
        
        return res;
            
    }
}
```

`optimal solution`

第二种解法产生了大量的函数调用 在函数的调用中 压栈出栈需要消耗性能

转换思路 用一个一维数组保存当前节点的决策

0 不偷当前节点 total =  左孩子能偷到的钱数 + 右孩子能偷到的钱数

1 偷当前节点 total = 当前节点能偷到的钱数 + 左孩子选择不偷能得到的钱 + 右孩子选择不偷能得到的钱 

> root[0] = Math.max(rob(root.left)[0], rob(root.left)[1]) + Math.max(rob(root.right)[0], rob(root.right)[1])
>
> root[1] = rob(root.left)[0] + rob(root.right)[0] + root.val

```java
class Solution {
    public int rob(TreeNode root) {
        int[] res = robInternal(root);
        return Math.max(res[0], res[1]);
    }
    private int[] robInternal(TreeNode root) {
        if (root == null) return new int[2];
        int[] res = new int[2];
        int[] left = robInternal(root.left);
        int[] right = robInternal(root.right);
        res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        res[1] = root.val + left[0] + right[0];
        return res;
    }
}
```



---

***best time to buy and sell stock problem 一个状态方程团灭6个股票问题***

*状态机* 实质上就是DP table

利用<u>**状态**</u>进行穷举，我们具体到每一天，看看总共有几种可能的<u>**状态**</u>，再找出每个**<u>状态</u>**对应的**<u>选择</u>**。

```
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 择优(选择1，选择2...)
```

股票问题的状态转移方程及base case

```
base case：
dp[-1][k][0] = dp[i][0][0] = 0
dp[-1][k][1] = dp[i][0][1] = -infinity

状态转移方程：
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
```

[reference]: https://github.com/labuladong/fucking-algorithm/blob/master/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%B3%BB%E5%88%97/%E5%9B%A2%E7%81%AD%E8%82%A1%E7%A5%A8%E9%97%AE%E9%A2%98.md

## best time to buy and sell stock

只使用两个变量来记录相邻的状态 完整使用二维数组存的（K = 1 || K = infinite 对状态转移无影响 所以不穷举，同时注意 k = 0 的 base case 所以dp [ i - 1] [0] [0] = 0 后面解法均采用优化的dp（注意看清实质）

```
dp[i][1][1] = max(dp[i-1][1][1], dp[i-1][0][0] - prices[i]) 
            = max(dp[i-1][1][1], -prices[i])
```

`optimal dp`  

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

## best time to buy and sell stock II

`optimal dp`

```java
class Solution {
    public int maxProfit(int[] prices) {
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        for (int i = 0; i < prices.length; i++) {
            int temp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, temp - prices[i]);
        }
        return dp_i_0;
    }
}
```

## best time to buy and sell stock with cooldown

`optimal dp`

```java
class Soliution {
	public int maxProfit(int[] prices) {
		int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        int dp_prev_0 = 0;
        for (int i = 0; i < prices.length; i++) {
            int temp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, dp_prev_0 - prices[i]);
            dp_prev_0 = temp;
        }
        return dp_i_0;
    }
}
```

## best time to buy and sell stock with transaction fee

`optimal dp`

```java
class Solution {
    public int maxProfit(int[] prices, int fee) {
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        for (int i = 0; i < prices.length; i++) {
			int temp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, temp - prices[i] - fee);
        }
        return dp_i_0;
    }
}
```

## best time to buy and sell stock III

`dp`

```java
class Solution {
    public int maxProfit(int[] prices) {
        int max_k = 2;
        int[][][] dp = new int[prices.length][max_k + 1][2];
        for (int i = 0; i < prices.length; i++) {
            for (int k = max_k; k >= 1; k--) {
                if (i == 0) {
                    dp[0][k][0] = 0;
                    dp[0][k][1] = -prices[0];
                    continue;
                }
                dp[i][k][0] = Math.max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i]);
                dp[i][k][1] = Math.max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
            }
        }
        return dp[prices.length - 1][max_k][0];
    }
}
```

`optimal dp`

```java
class Solution {
    public int maxProfit(int[] prices) {     
        int dp_i10 = 0, dp_i11 = Integer.MIN_VALUE;
        int dp_i20 = 0, dp_i21 = Integer.MIN_VALUE;
        for (int price : prices) {
            dp_i20 = Math.max(dp_i20, dp_i21 + price);
            dp_i21 = Math.max(dp_i21, dp_i10 - price);
            dp_i10 = Math.max(dp_i10, dp_i11 + price);
            dp_i11 = Math.max(dp_i11, -price);
        }
        return dp_i20;
    }
}
```

## best time to sell and buy stock IV

有了上一题 k = 2 的铺垫，这题应该和上一题的第一个解法没啥区别。但是出现了一个超内存的错误，原来是传入的 k 值会非常大，dp 数组太大了。现在想想，交易次数 k 最多有多大呢？

一次交易由买入和卖出构成，至少需要两天。所以说有效的限制 k 应该不超过 n/2，如果超过，就没有约束作用了，相当于 k = +infinity。这种情况是之前解决过的。

直接把之前的代码重用：

```java
class Solution {
    public int maxProfit(int max_k, int[] prices) {
        int n = prices.length;
        if (max_k > n >> 1) return infinityMaxProfit(prices);

        int[][][] dp = new int[n][max_k + 1][2];
        for (int i = 0; i < n; i++) {
            for (int k = max_k; k >= 1; k--) {
                if (i - 1 == -1){
                    dp[i][k][0] = 0;
                    dp[i][k][1] = -prices[i];
                    continue;
                }
                dp[i][k][0] = Math.max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i]);
                dp[i][k][1] = Math.max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
            }
        }
        return dp[n - 1][max_k][0];
    }
    private int infinityMaxProfit(int[] prices) {
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        for (int p : prices) {
            int temp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + p);
            dp_i_1 = Math.max(dp_i_1, temp - p);
        }
        return dp_i_0;
    }
}
```

`optimal dp`

```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        int len = prices.length;
        if (len < 2 || k == 0) {
            return 0;
        }         
        if (k > len / 2) {
            return greedy(prices);
        }
        // dp[i][j][0(1)] 表示第i + 1天交易了j次以后，不持有(持有)股票可以获得的最大收益
        int[][] dp = new int[k + 1][2];
        
        // 初始化
        for (int i = 0; i <= len; i++) {
            for (int j = 0; j <= k; j++) {
                dp[j][1] = Integer.MIN_VALUE;
            }
        }
        
        // 开始填充dp矩阵
        for (int i = 1; i <= len; i++) {
            for (int j = 1; j <= k; j++) {
                dp[j][1] = Math.max(dp[j][1], dp[j - 1][0] - prices[i - 1]);
                dp[j][0] = Math.max(dp[j][0], dp[j][1] + prices[i - 1]);
            }
        }        
        return dp[k][0];
    }
    
    private int greedy(int[] prices) {
        int res = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1])
            res += prices[i] - prices[i - 1];
        }
        return res;
    }
}
```

## binary tree cameras

`贪心`

自底向上处理逻辑 三种状态表示当前节点

+ 0 无覆盖
+ 1 安装摄像头
+ 2 有覆盖

其中 空节点表示为已覆盖 因为这样叶子节点不需要考虑空节点 如果标记为无覆盖 那么叶子节点就必须要加摄像头了

之所以是贪心 就是通过判断左右孩子的状态来确定当前节点的状态

+ 左右孩子只要有一个无覆盖 那么该节点就加摄像头 res++ 状态返回 1
+ 左右孩子只要有一个摄像头 那么该节点就不需要加摄像头 状态返回 2
+ 最后一种情况 左右孩子都是已覆盖 那么 当前节点需要被父节点覆盖 暂时标记为未覆盖（贪心） 状态返回 0

因为是贪心 最后一种情况可能会出现根节点的左右孩子状态为2 这样根节点就会返回 0 但是 根节点没有父节点帮他覆盖了 所以最后要判断一下 根节点返回状态 == 0 ? ++res : res;

+ 如果为0 那么根节点要设置摄像头

+ 否则 直接返回res

```java
class Solution {
    int res=0;
    public int minCameraCover(TreeNode root) {       
       return  dfs(root)==0?res+1:res;//如果父节点是无覆盖状态，那么需要在父节点添加一台摄像机
    }
    public int dfs(TreeNode root){
        if(root==null)
           return 2;//节点有覆盖
        int left=dfs(root.left);
        int right=dfs(root.right);
        //0表示无覆盖，1表示有相机，2表示有覆盖，左右节点可以组成9种状态
        //(00,01,02,10,11,12,20,21,22)
               
        //只要有一个无覆盖，父节点就需要相机来覆盖这个子节点 00,01,10,20,02
        if(left==0||right==0){
            res++;
            return 1;
        }
         //子节点其中只要有一个有相机，那么父节点就会是有覆盖的状态11,21,12
        if(left==1||right==1){
            return 2;
        }
        //还有一个就是22，两个子节点都是有覆盖的状态，父节点可以没有相机，可以借助它自己的父节点
        return 0;
    }
}
```

## implement trie prefix tree

```java
class Trie {
    class TrieNode {
        TrieNode[] next;
        boolean isEnd;
        public TrieNode() {
            next = new TrieNode[26];
            isEnd = false;
        }
    }
    
    TrieNode root;
    /** Initialize your data structure here. */
    public Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode node = root;
        for (char ch : word.toCharArray()) {
            if (node.next[ch - 'a'] == null) {
                node.next[ch - 'a'] = new TrieNode();
            }
            node = node.next[ch - 'a'];
        }
        node.isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode node = root;
        for (char ch : word.toCharArray()) {
            if (node.next[ch - 'a'] == null) return false;
            node = node.next[ch - 'a'];
        }
        return node.isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char ch : prefix.toCharArray()) {
            if (node.next[ch - 'a'] == null) return false;
            node = node.next[ch - 'a'];
        }
        return true;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
```

## implement stack using queues

两个队列 每次push压栈操作将原队列倒置 然后交换引用

```java
class MyStack {
    Queue<Integer> q1;
    Queue<Integer> q2;
    /** Initialize your data structure here. */
    public MyStack() {
        q1 = new LinkedList<>();
        q2 = new LinkedList<>();
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        q2.add(x);
        while (!q1.isEmpty()) q2.add(q1.poll());
        Queue<Integer> temp;
        temp = q1;
        q1 = q2;
        q2 = temp;
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        return q1.poll();
    }
    
    /** Get the top element. */
    public int top() {
        return q1.peek();
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return q1.isEmpty();
    }
}

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * boolean param_4 = obj.empty();
 */
```

## implement queue using stacks

pop()和peek()方法 都需要置换 所以置换的方法抽象出来

```java
class MyQueue {
	Stack<Integer> s1, s2;
    /** Initialize your data structure here. */
    public MyQueue() {
		s1 = new Stack<>();
        s2 = new Stack<>();
    }
    
    /** Push element x to the back of queue. */
    public void push(int x) {
		s1.push(x);
    }
    private void reverse() {
        if (s2.isEmpty()) 
            while (!s1.isEmpty()) s2.push(s1.pop());
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
		reverse();
        return s2.pop();
    }
    
    /** Get the front element. */
    public int peek() {
		reverse();
        return s2.peek();
    }
    
    /** Returns whether the queue is empty. */
    public boolean empty() {
		return s1.isEmpty() && s2.isEmpty();
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * boolean param_4 = obj.empty();
 */
```



## generate parentheses

`剪枝`

```java
class Solution {
    List<String> list;
    public List<String> generateParenthesis(int n) {
    	list = new ArrayList<>();
        _generator(0, 0, n, "");
        return list;
    }
    public void _generator(int left, int right, int max, String s) {
        if (left == max && right == max) {
            list.add(s);
            return
        }
        if (left < max) _generator(left + 1, right, max, s + "(");
        if (left > right) _generator(left, right + 1, max, s + ")");
    }
}
```

## valid sudoku

`2D array`

方块索引 = (i / 3) * 3 + j / 3

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        int[][] row = new int[9][10];
        int[][] col = new int[9][10];
        int[][] box = new int[9][10];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == '.') continue;
                int currNum = board[i][j] - '0';
                if (row[i][currNum] == 1) return false;
                if (col[j][currNum] == 1) return false;
                if (box[j / 3 + (i / 3) * 3][currNum] == 1) return false;
                row[i][currNum] = 1;
                col[j][currNum] = 1;
                box[j / 3 + (i / 3) * 3][currNum] = 1;
            }
        }
        return true;
    }
}
```

## sudoku solver

```java
class Solution {
    public void solveSudoku(char[][] board) {
        if(board == null || board.length == 0) return;
        solve(board);
    }
    public boolean solve(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == '.') {
                    for (char c = '1'; c <= '9'; c++) {
                        if (isValid(board, i, j, c)) {
                            board[i][j] = c;
                            if (solve(board)) return true;
                            board[i][j] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }
    public boolean isValid (char[][] board, int row, int col, char c) {
        for (int i = 0; i < 9; i++) {
            //check colomn
            if (board[i][col] != '.' && board[i][col] == c) return false;
            if (board[row][i] != '.' && board[row][i] == c) return false;
            if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] != '.' &&
            board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c ) return false;
        }
        return true;
    }

}
```

## shortest path in binary matrix

`BFS`

`A* Search`

## sliding puzzle

`BFS`

`A* Search`

## number of 1 bits 

`每一位与掩码 & 如果不为0 说明该位为1 时间复杂度 O（1） 执行32次`

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

`利用n & n-1 将最低位的1变为零 每执行一次操作num ++直到n变为0 执行1的个数次`

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

## hamming distance

`汉明距离`

基于《number of 1 bits 》题目求汉明重量的学习
先求两个数的异或（每一位相同为0，不同为1）
然后通过求这个异或的汉明重量即为这两个数的汉明距离

```java
class Solution {
    public int hammingDistance(int x, int y) {
        int n = x ^ y;
        int num = 0;
        while (n != 0) {
            num++;
            n &= (n - 1);
        }
        return num;
    }
    
}
```

## power of two 

`2的幂二进制下汉明重量为1 `

**汉明重量**是一串符号中非零符号的个数

```java
class Solution {
    public int hammingWeight(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}
```

## reverse bits 

```java
class Solution {
    public int reverseBits(int n) {
        int ans = 0;
        for (int i = 0; i < 32; i++) {
            ans = (ans << 1) + (n & 1);
            n = n >> 1;
        }
        return ans;
    }
}
```

## counting bits

`dp + 位运算`

```java
class Solution {
	public int[] countBits(int num) {
		int[] dp = new int[num + 1];
        for (int i = 1; i <= num; i++) {
            dp[i] = dp[i & (i - 1)] + 1;
        }
        return dp; 
    }
}
```



## n-queens

`backtracking`

```java
class Solution {
    List<List<String>> result = new ArrayList<>();
    public List<List<String>> solveNQueens(int n) {
        boolean[] visited = new boolean[n];
        //2*n-1个斜对角线
        boolean[] dia1 = new boolean[2*n-1];
        boolean[] dia2 = new boolean[2*n-1];
        
        fun(n, new ArrayList<String>(),visited,dia1,dia2,0);
        
        return result;
    }
    
    private void fun(int n,List<String> list,boolean[] visited,boolean[] dia1,boolean[] dia2,int rowIndex){
        if(rowIndex == n){
            result.add(new ArrayList<String>(list));
            return;
        }
        
        for(int i=0;i<n;i++){
            //这一行、正对角线、反对角线都不能再放了，如果发现是true，停止本次循环
            if(visited[i] || dia1[rowIndex+i] || dia2[rowIndex-i+n-1])
                continue;
            
            //init一个长度为n的一维数组，里面初始化为'.'
            char[] charArray = new char[n];
            Arrays.fill(charArray,'.');
            
            charArray[i] = 'Q';
            String stringArray = new String(charArray);
            list.add(stringArray);
            visited[i] = true;
            dia1[rowIndex+i] = true;
            dia2[rowIndex-i+n-1] = true;

            fun(n,list,visited,dia1,dia2,rowIndex+1);

            //reset 不影响回溯的下个目标
            list.remove(list.size()-1);
            charArray[i] = '.';
            visited[i] = false;
            dia1[rowIndex+i] = false;
            dia2[rowIndex-i+n-1] = false;
        }
    }
}
```

`位运算`

```java
class Solution {
    private int size;
    private int count;
    
    private void solve(int row, int ld, int rd) {
        if (row == size) {
            count ++;
            return;
        }
        int pos = size & (~(row | ld | rd));
        while (pos != 0) {
            int p = pos & (-pos);
            pos -= p; //pos &= pos - 1;
            solve(row | p, (ld | p) << 1, (rd | p) >> 1);
        }
    }
    
    public int totalNQueens(int n) {
        count = 0;
        size = (1 << n) - 1;
        solve(0, 0, 0);
        return count;
    }
}
```

## n-queens ii

## LRU cache

`hash + 双向链表` 实现查询O1（get操作），添加删除O1（put操作）

https://leetcode-cn.com/problems/lru-cache/solution/lru-ce-lue-xiang-jie-he-shi-xian-by-labuladong/

+ 为什么需要双向链表？单链表不行吗

因为删除操作不仅要得到当前节点 还要去操作指向该节点的前驱节点 所以要用双向链表直接获得该节点的前驱节点

+ hash表已经存key 了 为什么双向链表还要保存key value键值对？





```java
class LRUCache {
    private class Node {
        Node prev, next;
        int key, value;

        private Node(int k, int v) {
            this.key = k;
            this.value = v;
        }
    }

    private class DoubleList {
        Node head = new Node(0, 0);
        Node tail = new Node(0, 0);
        int size;

        private DoubleList() {
            head.next = tail;
            tail.prev = head;
            size = 0;
        }

        private void addFirst(Node n) {
            Node headNext = head.next;
            head.next = n;
            headNext.prev = n;
            n.prev = head;
            n.next = headNext;
            size++;
        }

        private void remove(Node n) {
            n.prev.next = n.next;
            n.next.prev = n.prev;
            size--;
        }

        private Node removeLast() {
            Node last = tail.prev;
            remove(last);
            return last;
        }

        private int size() {
            return size;
        }

    }

    // key -> Node(key, val)
    private Map<Integer, Node> map;
    // node(k1, v1) <-> Node(k2, v2)...
    private DoubleList cache;
    private int capacity;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        map = new HashMap<>(capacity);
        cache = new DoubleList();
    }

    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        }
        int val = map.get(key).value; // 利⽤ put ⽅法把该数据提前
        put(key, val);
        return val;
    }

    public void put(int key, int value) {
        Node n = new Node(key, value);
        if (map.containsKey(key)) {
            cache.remove(map.get(key));
            cache.addFirst(n);
            map.put(key, n);
        } else {
            if (cache.size() == capacity) {
                // delete last element in list
                Node last = cache.removeLast();
                map.remove(last.key);
            }
            cache.addFirst(n);
            map.put(key, n);
        }
    }
}
```



## relative sort array

`计数排序`

```java
class Solution {
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        int[] arr = new int[1001];
        int[] res = new int[arr1.length];
        int idx = 0;
        for (int num : arr1) {
            arr[num] ++;
        }
        for (int num : arr2) {
            while (arr[num] -- > 0) {
                res[idx] = num;
                idx ++;
            }
        }
        for (int i = 0; i < 1001; i++) {
            while (arr[i] -- > 0) {
                res[idx] = i;
                idx ++;
            }
        }
        
        return res;
    }
}
```

## merge intervals

```java
class Solution {
    public int[][] merge(int[][] intervals) {
		if (intervals.length <= 1) return intervals;
        List<int[]> res = new ArrayList<>();
        Arrays.sort(intervals, (a, b) -> (a[0]- b[0])); //重要 排序保证每个左区间单调
        int start = intervals[0][0];
        int end = intervals[0][1];
        for (int[] i : intervals) {
			if (i[0] <= end) end = Math.max(i[1], end);
            else {

                res.add(new int[]{start, end});
                start = i[0];
                end = i[1];
                
            }
        }
        //如果最后一个区间要合并 则会更新end 如果不需要合并 此时添加的是上一个合并的区间
        //所以最后都要再添加最后一个区间
         res.add(new int[]{start, end});
        return res.toArray(new int[0][]);
    }
}
```

## reverse pairs

逆序对 问题

best solution 1.mergeSort 2.树状数组 时间复杂度 O(nlogn)

`merge`

```java
class Solution {
 	public int reversePairs(int[]nums) {
        if (nums == null || nums.length == 0) return 0;
        return merge(nums, 0, nums.length - 1);
    }   
    public int merge(int[] arr, int left, int right) {
		if (right <= left) return 0;
        int mid = (right + left) >> 1;
        int cnt = merge(arr, left, mid) + merge(arr, mid + 1, right);
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, k = 0;
        int p = mid + 1;
        while (i <= mid) {
            //count reverse pair
			while (p <= right && arr[i] > 2L * arr[p]) p++;
            cnt += p - (mid + 1);
            //sort and merge
            while (j <= right && arr[i] > arr[j]) temp[k++] = arr[j++];
            temp[k++] = arr[i++];
        }
        while(j <= right) temp[k++] = arr[j++];
        System.arraycopy(temp, 0, arr, left, temp.length);
        return cnt;
    }
}
```

## edit distance

`DP`

![image-20210420153745905](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210420153745905.png)

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int i = 1; i <= n; i++) dp[0][i] = i;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    int a = dp[i - 1][j];
                    int b = dp[i][j - 1];
                    int c = dp[i - 1][j - 1];
                    dp[i][j] = a < b ? (a < c ? a : c) : (b < c ? b : c);
                    dp[i][j]++;
                }
            }
        }
        return dp[m][n];
    }
}
```

## longest increasing subsequence

`dp` o(n*n) 

```java
 public static int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            res = Math.max(res, dp[i]);
        }
        for (int i : dp) {
            System.out.print(i + " ");
        }
        return res;
    }
```

dp + 二分法 O(NlogN)

```java
public static int lengthOfLIS(int[] nums) {
    int[] tails = new int[nums.length];
    int res = 0;
    for(int num : nums) {
        int i = 0, j = res;
        while(i < j) {
            int m = (i + j) / 2;
            if(tails[m] < num) i = m + 1;
            else j = m;
        }
        tails[i] = num;
        if(res == j) res++;
    }
    return res;
}
```

## decode ways

类似于jcof将数字翻译成字符串，不过情况更复杂

## longest valid parentheses

`dp`

```java
public class Solution {
    public int longestValidParentheses(String s) {
        int maxans = 0;
        int[] dp = new int[s.length()];
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = Math.max(maxans, dp[i]);
            }
        }
        return maxans;
    }
}
```

`stack`

核心思路 栈中始终维持最后一个 未匹配的右括号的下标 (初始化要放入-1 否则一开始左括号放入不符合)

 利用栈 如果是左括号就放入下标 当遇到右括号就弹出栈顶元素 表示该右括号已匹配 否则根据栈内是否为空

若为空则弹入最后一个右括号的下标 表示该括号未被匹配 若不为空则更新maxLen 与 当前字符串长度i - stack.peek()栈顶元素 即最后一个不匹配的右括号的下标

```java
class Solution {
    public int longestValidParentheses(String s) {
        int maxLen = 0;
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') stack.push(i);
            else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else {
                    maxLen = Math.max(maxLen, i - stack.peek());
                }
            }

        }
        return maxLen;
    }
}
```



## maximal rectangle

类比 largset rectangle in histogram 84, 每一层得到数组的高度，然后去计算最大的矩形面积，需要调用`maxrix.length`次 不是最优解

```java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0) return 0;
        int res = 0;
        int[] rectangle = new int[matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == '1') rectangle[j] += 1; //更新高度
                else rectangle[j] = 0; //处理断层
            }
            res = Math.max(res, largestRectangleArea(rectangle));
        }
        return res;
    }
    public int largestRectangleArea(int[] heights) {
        int ans = 0;
        LinkedList<Integer> stack = new LinkedList<>();
        for (int i = 0; i <= heights.length; ) {
            int h = i == heights.length ? 0 : heights[i];
            if (stack.isEmpty() || h >= heights[stack.peekLast()]) stack.addLast(i++);
            else {
                int currHeight = heights[stack.pollLast()];
                int left = stack.isEmpty() ? 0 : stack.peekLast() + 1;
                int right = i - 1;
                ans = Math.max(ans, currHeight * (right - left + 1));
            }
        }
        return ans;
    }
}
```

`dp` 有时间补个dp 

## distinct subsequences

## race car

## string to integer atoi

```java
class Solution {
    public int myAtoi(String str) {
		int idx = 0, total = 0, sign = 1;
        if (str.length() == 0) return 0;
        while (idx < str.length() && str.charAt(idx) == ' ') idx++;
        if (idx < str.length() && (str.charAt(idx) == '+' || str.charAt(idx) == '-')) {
            sign = str.charAt(idx) == '+' ? 1 : -1;
            idx ++;
        }
        while (idx < str.length()) {
            int digit = str.charAt(idx) - '0';
            if (digit < 0 || digit > 9) break;
         	if (Integer.MAX_VALUE / 10 < total 
                || Integer.MAX_VALUE == total && Integer.MAX_VALUE % 10 < digit) 
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }
        total = 10 * total + digit;
        idx ++;
    }
    return total * sign;
}
```

## longest common prefix

```java
class Solution {
    public String longestCommonPrefix(String[] str) {
        if (str == null || str.length == 0) return "";
        for (int i = 0; i < str[0].length(); i++) {
            char c = str[0].charAt(i);
            for (int j = 1; j < str.length; j++) {
                if (i == str[j].length() || c != str[j].charAt(i)) return str[0].substring(0, i);
            }
        }
        return str[0];
    }
}
```

## reverse  string II

```java
class Solution {
    public String reverseStr(String s, int k) {
        char[] a = s.toCharArray();
        for (int start = 0; start < a.length; start += 2 * k) {
            int i = start, j = Math.min(start + k - 1, a.length - 1);
            while (i < j) {
                char temp = a[i];
                a[i++] = a[j];
                a[j++] = temp;
            }
        }
        return new String(a)
    }
}
```

## reverse words in a string

`正则表达式匹配多个空格字符` `//s+`

`集合元素reverse` `Collection.reverse(List<?> list)`

`String.join(delimiter:, elements:)`

```java
class Solution{
	public String reverseWords(String s) {
        s = s.trim();
        List<String> list = Arrays.asList(s.split("\\s+"));
        Collections.reverse(list);
        return String.join(" ", list);
    }
}
```

`API Recursion` fast and concise code

```java
class Solution {
   public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        getReverseWords(s, sb, 0, false);
        return sb.toString();
    }

    private void getReverseWords(String s, StringBuilder sb, int start, boolean flag) {
        // 跳过空格
        while (start < s.length() && s.charAt(start) == ' '){
            start ++;
        }
        if (start == s.length()){
            return;
        }

        // 获取单词，并递归
        int end = start;
        while (end < s.length() && s.charAt(end) != ' '){
            end ++;
        }
        getReverseWords(s, sb, end, true);

        // 递归回来后，插入本位置对应单词
        sb.append(s.substring(start, end));

        // flag为true时，插入单词后再插入空格
        if (flag) sb.append(" ");
    }
}
```



## group anagrams

`use key string instead of sort array`

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
	    if (strs == null || strs.length == 0) return new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] c = new char[26];
            for (char ch : s.toCharArray()) c[ch - 'a']++;
            String key = String.valueOf(c);
            map.putIfAbsent(key, new ArrayList<>());
            map.get(key).add(s);
        }
        return new ArrayList<>(map.values());
    }
}
```

## find all anagram in a string

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        int[] map = new int[26];
        int start = 0, end = 0;
        List<Integer> res = new ArrayList<>();
        for (char c : p.toCharArray()) map[c - 'a']++;
        while (end < s.length()) {
            //valid anagram
            if (map[s.charAt(end) - 'a'] > 0) {
                map[s.charAt(end++) - 'a'] --;
                //window size equals to size of p
                if (end - start == p.length()) res.add(start);
            } else if (end == start) {//window size  == 0
        		start++;
                end++;
            } else {
				map[s.charAt(start++) - 'a']++;
            }
        }
        return res;
    }
}
```

## longest palindrome

```java
class Solution {
    public int longestPalindrome(String s) {
        int[] c = new int[128];
        int count = 0;
        for (char ch : s.toCharArray()) {
			c[ch]++;
        }
        for (int ch : c) {
			count += (ch % 2);
        }
  		return count == 0 ? s.length() : s.length() - count + 1;      
    }
}
```

## longest palindromic substring

`solution`

Iterate the string, for each character, try to expand left and right to get the longest palindromic substring

```java
class Solution {
	public String longestPalindrome(String s) {
        if (s == null || s.trim().equals("")) {
			return s;
        }
        int len = s.length(), begin = 0, maxLen = 0;
        for (int i = 0; i < len - (maxLen >> 1); i++) {
            int j = i, k = i;
            //skip duplicated characters to the right
            while (k < len - 1 && s.charAt(k) == s.charAt(k + 1)) k++;
            //expand both left and right
			while (j > 0 && k < len - 1 && s.charAt(j - 1) == s.charAt(k + 1)) {
                j--; k++;
            }
            int newLen = k - j + 1;
            if (newLen > maxLen) {
                begin = j;
                maxLen = newLen;
            }
        }
        return s.substring(begin, begin + maxLen);
    }
}
```

## regular expression matching submissions

`dp`

![image-20210304211647145](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210304211647145.png)

```java
class Solution {
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') 
                    f[i][j] = f[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }

    public boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }
}
```

## wildcard matching

`dp`

![image-20210305131506365](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210305131506365.png)

```java
class Solution {
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; ++i) {
            if (p.charAt(i - 1) == '*') {
                dp[0][i] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                } else if (p.charAt(j - 1) == '?' || s.charAt(i - 1) == p.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }
}

```

## spiral matrix i

模拟法，设定边界

```java
class Solution {
 	public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> list = new ArrayList<>();
        int l = 0, r = matrix[0].length - 1, t = 0, b = matrix.length - 1;
        int tar = (r + 1) * (b + 1);
        while (tar >= 1) {
            //left to right t++
            for (int i = l; i <= r && tar-- >= 1; i++) list.add(matrix[t][i]); 
            t++;
            //top to bootom r--
            for (int i = t; i <= b && tar-- >= 1; i++) list.add(matrix[i][r]); 
            r--;
            //right to left b--
            for (int i = r; i >= l && tar-- >= 1; i--) list.add(matrix[b][i]);
            b--;
            //bottom to top l++
            for (int i = b; i >= t && tar-- >= 1; i--) list.add(matrix[i][l]);
            l++;
        }
        return list;
    }   
}
```



## spiral matrix ii

模拟法，设定边界

```java
class Solution {
    public int[][] generateMatrix(int n) {
        int l = 0, r = n - 1, t = 0, b = n - 1;
        int[][] mat = new int[n][n];
        int num = 1, tar = n * n;
        while (num <= tar) {
            for (int i = l; i <= r; i++) mat[t][i] = num++;//left to right
            t++; 
            for (int i = t; i <= b; i++) mat[i][r] = num++; //top to bottom
            r--;
            for (int i = r; i >= l; i--) mat[b][i] = num++; //right to left
            b--;
            for (int i = b; i >= t; i--) mat[i][l] = num++; //bottom to top
            l++;
        }
        return mat;
    }
}
```

## Linked list random node

`蓄水池算法`https://www.jianshu.com/p/7a9ea6ece2af

```java
public class Solution {
    
    ListNode head;
    Random random;
    
    public Solution(ListNode h) {
        head = h;       
        random = new Random();        
    }
    
    public int getRandom() {
        
        ListNode c = head;
        int r = c.val;
        for(int i=1;c.next != null;i++){
            
            c = c.next;
            if(random.nextInt(i + 1) == i) r = c.val;                        
        }
        
        return r;
    }
}
```

## median of two sorted arrays

`O(log(m, n)) time`

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
            int idx1 = 0, idx2 = 0;
            int med1 = 0, med2 = 0;
            for (int i = 0; i <= nums1.length + nums2.length >> 1; i++) {
                med1 = med2;
                if (idx1 == nums1.length) {
                    med2 = nums2[idx2];
                    idx2++;
                } else if (idx2 == nums2.length) {
                    med2 = nums1[idx1];
                    idx1++;
                } else if (nums1[idx1] < nums2[idx2]) {
                    med2 = nums1[idx1];
                    idx1++;
                } else {
                    med2 = nums2[idx2];
                    idx2++;
                }
            }
            //the median is the average of two numbers
            if ((nums1.length + nums2.length) % 2 == 0)  return (float)(med1+med2)/2;
            return med2;
        }
}
```

## perfect squares

`dp`类似coin change

```java
class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n + 1];//init 0
        for (int i = 1; i <= n; i++) {
            dp[i] = i;//worst condition every move plus one
            for (int j = 1; i - j * j >= 0; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);//动态转移方程
            }
        }
        return dp[n];
    }
}
```

`四平方和定理`

```java
public int numSquares(int n) {
    //判断是否是 1
    if (isSquare(n)) {
        return 1;
    }
    
    //判断是否是 4
    int temp = n;
    while (temp % 4 == 0) {
        temp /= 4;
    }
    if (temp % 8 == 7) {
        return 4;
    }

    //判断是否是 2
    for (int i = 1; i * i < n; i++) {
        if (isSquare(n - i * i)) {
            return 2;
        }
    }

    return 3;
}

//判断是否是平方数
private boolean isSquare(int n) {
    int sqrt = (int) Math.sqrt(n);
    return sqrt * sqrt == n;
}
```

## smallest good base

`math`

```java
class Solution {
    public String smallestGoodBase(String n) {
        long nVal = Long.parseLong(n);
        int mMax = (int) Math.floor(Math.log(nVal) / Math.log(2));
        for (int m = mMax; m > 1; m--) {
            int k = (int) Math.pow(nVal, 1.0 / m);
            long mul = 1, sum = 1;
            for (int i = 0; i < m; i++) {
                mul *= k;
                sum += mul;
            }
            if (sum == nVal) {
                return Integer.toString(k);
            }
        }
        return Long.toString(nVal - 1);
    }
}
```



## largest numer

类比jzof 45.把数组排成最小的数  本质是快排

```java
class Solution {
    public String largestNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for (int i = 0; i < nums.length; i++) strs[i] = String.valueOf(nums[i]);
        quickSort(strs, 0, strs.length - 1);
        if (strs[0].equals("0")) {
            return "0";
        } else {
            StringBuilder res = new StringBuilder();
            for (String s : strs) res.append(s);
            return res.toString();
        }
    }
    void quickSort(String[] strs, int l, int r) {
         if (r <= l) return;
        int i = l, j = r;
        String temp = strs[i];
        while (i < j) {
            //以l作为pivot
            while (i < j && (strs[j] + strs[l]).compareTo(strs[l] + strs[j]) <= 0) j--;
            while (i < j && (strs[i] + strs[l]).compareTo(strs[l] + strs[i]) >= 0) i++;
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

## split array largest sum

`二分`

```java
public class Solution {

    public int splitArray(int[] nums, int m) {
        int max = 0;
        int sum = 0;

        // 计算「子数组各自的和的最大值」的上下界
        for (int num : nums) {
            max = Math.max(max, num);
            sum += num;
        }

        // 使用「二分查找」确定一个恰当的「子数组各自的和的最大值」，
        // 使得它对应的「子数组的分割数」恰好等于 m
        int left = max;
        int right = sum;
        while (left < right) {
            int mid = left + (right - left) / 2;

            int splits = split(nums, mid);
            if (splits > m) {
                // 如果分割数太多，说明「子数组各自的和的最大值」太小，此时需要将「子数组各自的和的最大值」调大
                // 下一轮搜索的区间是 [mid + 1, right]
                left = mid + 1;
            } else {
                // 下一轮搜索的区间是上一轮的反面区间 [left, mid]
                right = mid;
            }
        }
        return left;
    }

    /***
     *
     * @param nums 原始数组
     * @param maxIntervalSum 子数组各自的和的最大值
     * @return 满足不超过「子数组各自的和的最大值」的分割数
     */
    private int split(int[] nums, int maxIntervalSum) {
        // 至少是一个分割
        int splits = 1;
        // 当前区间的和
        int curIntervalSum = 0;
        for (int num : nums) {
            // 尝试加上当前遍历的这个数，如果加上去超过了「子数组各自的和的最大值」，就不加这个数，另起炉灶
            if (curIntervalSum + num > maxIntervalSum) {
                curIntervalSum = 0;
                splits++;
            }
            curIntervalSum += num;
        }
        return splits;
    }
}

```

## remove duplicates from sorted list

`iteration`

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode cur = head;
        while(cur != null && cur.next != null) {
            if(cur.val == cur.next.val) {
                cur.next = cur.next.next;
            } else {
                cur = cur.next;
            }
        }
        return head;
    }
}
```

## remove duplicates from sorted list ii

`iteration`

```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return head;
        }        
        ListNode dummy = new ListNode(0, head);
        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                int x = cur.next.val;
                while (cur.next != null && cur.next.val == x) {
                    cur.next = cur.next.next;
                }
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }
}
```

## valid mountain array

一个从左边找最高山峰，一个从右边找最高山峰，最后判断找到的是不是同一个山峰

```java
class Solution {
    public boolean validMountainArray(int[] arr) {
        int l = 0, r = arr.length - 1;
        while (l + 1 < arr.length && arr[l] < arr[l + 1]) l++;
        while (r > 0 && arr[r] < arr[r - 1]) r--;
        return l > 0 && r < arr.length - 1 && l == r;
    }
}
```

## longest mountain in array

`double pointer`枚举山脚

```java
class Solution {
    public int longestMountain(int[] arr) {
        int n = arr.length;
        int ans = 0;
        int left = 0;
        while (left + 2 < n) {
            int right = left + 1;
            if (arr[left] < arr[left + 1]) {
                while (right + 1 < n && arr[right] < arr[right + 1]) {
                    ++right;
                }//此时r指向山顶
                if (right < n - 1 && arr[right] > arr[right + 1]) {
                    while (right + 1 < n && arr[right] > arr[right + 1]) {
                        ++right;
                    }
                    ans = Math.max(ans, right - left + 1);
                } else {
                    ++right;
                }
            }
            left = right;
        }
        return ans;
    }
}
```

## subtree of another tree

`recusion`注意区分子树 和 子树结构（jzof 26)

```java
class Solution {
    public boolean isSubtree(TreeNode s, TreeNode t) {
        return (s != null && t != null) && (recur(s, t) || isSubtree(s.left, t) || isSubtree(s.right, t));
    }
    boolean recur (TreeNode s, TreeNode t) {
        if (s == null && t == null) return true;
        if (s == null || t == null || s.val != t.val) return false;
        return recur(s.left, t.left) && recur(s.right, t.right);
    }
}
```

## factorial trailing zeroes

`计算5的因子的个数`

```java
public int trailingZeroes(int n) {
    int count = 0;
    while (n > 0) {
        count += n / 5;
        n = n / 5;
    }
    return count;
}
```

## shuffle an array

` Fisher-Yates 洗牌算法`

Fisher-Yates 洗牌算法跟暴力算法很像。在每次迭代中，生成一个范围在当前下标到数组末尾元素下标之间的随机整数。接下来，将当前元素和随机选出的下标所指的元素互相交换 - 这一步模拟了每次从 “帽子” 里面摸一个元素的过程，其中选取下标范围的依据在于每个被摸出的元素都不可能再被摸出来了。此外还有一个需要注意的细节，当前元素是可以和它本身互相交换的 - 否则生成最后的排列组合的概率就不对了。

```java
class Solution {
    private int[] array;
    private int[] original;

    Random rand = new Random();

    private int randRange(int min, int max) {
        return rand.nextInt(max - min) + min;
    }

    private void swapAt(int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    public Solution(int[] nums) {
        array = nums;
        original = nums.clone();
    }
    
    public int[] reset() {
        array = original;
        original = original.clone();
        return original;
    }
    
    public int[] shuffle() {
        for (int i = 0; i < array.length; i++) {
            swapAt(i, randRange(i, array.length));
        }
        return array;
    }
}
```

## restore ip addresses

`backtracking`

```java
 public List<String> restoreIpAddresses(String s) {
        List<String> result = new ArrayList<>();
        doRestore(result, "", s, 0);
        return result;
    }
    
    private void doRestore(List<String> result, String path, String s, int k) {
        if (s.isEmpty() || k == 4) {
            if (s.isEmpty() && k == 4)
                result.add(path.substring(1));
            return;
        }
        for (int i = 1; i <= (s.charAt(0) == '0' ? 1 : 3) && i <= s.length(); i++) { // Avoid leading 0
            String part = s.substring(0, i);
            if (Integer.valueOf(part) <= 255)
                doRestore(result, path + "." + part, s.substring(i), k + 1);
        }
    }
```

## consercutive numbers sum

`brute violence`

```java
//滑动窗口
class Solution {
    public int consecutiveNumbersSum(int N) {
        int i = 1, j = 1, sum = 0;
        int cnt = 1;
        while (i < (N >> 1)) {
            if (sum > N) sum -= i++;
            else if (sum < N) sum += j++;
            else {
                cnt++;
                sum -= i++;
            }
        }
        return cnt++;
    }
}
```

`math`

```java
class Solution {
    public int consecutiveNumbersSum(int N) {
        while ((N & 1) == 0) N >>= 1;
        int ans = 1, d = 3;

        while (d * d <= N) {
            int e = 0;
            while (N % d == 0) {
                N /= d;
                e++;
            }
            ans *= e + 1;
            d += 2;
        }

        if (N > 1) ans <<= 1;
        return ans;
    }
}
```

## convert sorted list to binary search tree

```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
      if(head == null)return null;
      if(head.next == null)return new TreeNode(head.val);

      ListNode slow = head, fast = head, pre = head;
      while(fast != null && fast.next != null){
        pre = slow;
        slow = slow.next;
        fast = fast.next.next;
      }
      
      ListNode right = slow.next;
      pre.next = null;
      TreeNode root = new TreeNode(slow.val);
      root.left = sortedListToBST(head);
      root.right = sortedListToBST(right);

      return root;
    }   
}
```

## longest substring with at least k repeating characters

`recursion`

```java
class Solution {
    public int longestSubstring(String s, int k) {
        if (s.length() < k) return 0;
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        for (char c : map.keySet()) {
            if (map.get(c) < k) {
                int res = 0;
                for (String t : s.split(String.valueOf(c))) {
                    res = Math.max(res, longestSubstring(t, k));
                }
                return res;
            }
        }
        return s.length();
    }
}
```

## building boxes

规律

从 0 开始，与地面接触的盒子每增加一个，可以放置的总盒子数就会增加 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5 ... 个。

```java
class Solution {
    public int minimumBoxes(int n) {
        int bottom = 1;
        for (int sum = 1, height = 1; sum < n; height++) {
            for (int i = 0; i <= height && sum < n; i++) { // 每次增加 1 个底部盒子
                bottom++;
                sum += i + 1; // i 个是在上方堆叠的，不与地面接触
            }
        }
        return bottom;
    }
}
```

`maxTotal(i) = (i^3+3i^2+2i)/6 = i(i+1)(i+2)/6` 第i层 能放置的最大盒子数

```java
class Solution {
    //备注：由于java的基本类型数据范围限制，计算过程中必须要做防溢出处理
    public int minimumBoxes(int n) {
        //根据题意，分析可知：
        // 0.从上到下：第1层，第2层，第3层...能放置的最大盒子数分别是：1,3,6...
        // 1.从上到下第i层的最大盒子数maxLayer(i) = 1+2+...+i = i(i+1)/2
        // 2.从第1层到第i层的盒子总和maxTotal(i) = [(1^2+1) + (2^2+2) + ... + (i^2+i)]/2,
        //                                  = [(1+...+i) + (1^2+...+i^2)]/2
        //                                  = [i(i+1)/2 + i(i+1)(2i+1)/6]/2
        //  上述式子前半部分为等差数列求和公式Sn = n*a1 + n(n-1)*d/2，这里a1=1,d=1，所以可以化简为i(i+1)/2
        //  后半部分为自然数平方和公式1^2+...+n^2 = n(n+1)(2n+1)/6
        //  最终化简后，maxTotal(i) = (i^3+3i^2+2i)/6 = i(i+1)(i+2)/6

        //先二分查找小于等于n的最大完全堆的层高（即每一层都是能放置的极限值）
        int left=0, right=n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            //(mid * mid+1 * mid+2)/6 < n 的防溢出写法
            if ((double)mid * (mid + 1) / 6 < (double)n / (mid + 2)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        //上述循环打破的条件是left=right，而right是最后一个【大于n的最大完全堆的层高】，
        // 所以小于等于n的最大完全堆的层高应该是left-1
        int minHeight = left - 1;
        /*//此时可以认为，n个盒子中，maxTotal(left)个已被最佳放置，且层高为left
        //接下来需要处理剩下的盒子：找到最底层每增加一个盒子，整体最多增加多少个盒子的关系
        //当底层的盒子增加了k个，整体增加的最大值刚好大于等于剩余盒子数时，问题就解决了
        //以如下矩阵表示一个完全堆的状态，矩阵是完全堆的俯视图，矩阵中每个数字表示那个位置的盒子高度：
        //[[3,2,1],
        // [2,1,0],
        // [1,0,0]]
        //根据题意，要让盒子最佳放置，每次在一个完全堆基础上新增盒子，应该沿着最底层的对角线放置
        //在上述完全堆中第一次新增一个盒子，矩阵应该变成下面这样：
        //[[3,2,1,0],
        // [2,1,0,0],
        // [1,0,0,0],
        //【1,0,0,0】
        //第二次：
        //[[3,2,1,0],
        // [2,1,0,0],
        //【2,1,0,0】,
        // [1,0,0,0]
        //第三次：
        //[[3,2,1,0],
        //【3,2,1,0】,
        // [2,1,0,0],
        // [1,0,0,0]
        //第4次：
        //[【4,3,2,1】,
        //  [3,2,1,0],
        //  [2,1,0,0],
        //  [1,0,0,0]
        //于是最底层每增加1个，2个，3个...能容纳的新增盒子总数为1,3,6...
        //这个规律和每一层能放置的最大盒子数的计算方式是相同的：maxLayer(i) = 1+2+...+i = i(i+1)/2*/
        long minArea = maxLayer(minHeight);
        long remain = n - maxTotal(minHeight);
        left = 0; right = (int)remain;
        while (left < right) {
            int mid = (left + right) / 2;
            if (maxLayer(mid) < remain) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return (int)minArea + left;
    }

    //返回第i层的最大盒子数，防止溢出需要用long运算
    long maxLayer(int i){
        return (long)i*(i+1)/2;
    }

    //返回层高为i的完全堆的盒子总数
    long maxTotal(int i){
        return (long)i*(i+1)*(i+2)/6;
    }
}

```

## symmetric tree

`recur`

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return recur(root.left, root.right);
    }
    boolean recur(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
         if (left == null || right == null) return false;
         if (left.val != right.val) return false;
        return recur(left.left,right.right) && recur(right.left, left.right);
    }
}
```

`iteration`

```java
public boolean isSymmetric(TreeNode root) {
        Deque<TreeNode> queue = new LinkedList<>();
        queue.add(root.left);
        queue.add(root.right);
        while(!queue.isEmpty()) {
            TreeNode left = queue.poll();
            TreeNode right = queue.poll();
            if (left == null && right == null) continue;
            else if (left == null || right == null) return false;
            else if (left.val != right.val) return false;
            queue.add(left.left);
            queue.add(right.right);
            queue.add(left.right);
            queue.add(right.left); 
        }
        return true;
    }
```

## maximum repeating substring

```java
class Solution {
    public int maxRepeating(String sequence, String word) {
        int count = -1;
        StringBuilder sb = new StringBuilder();
        while (sequence.contains(sb)) {
            count++;
            sb.append(word);
        }
        return count;
    }
}
```

## find-first-and-last-position-of-element-in-sorted-array

`二分法`

实质是找到第一个小于target的元素和第一个大于target的元素

nums[m] >= target    => 左边界 因为所有值都大于target 此时指针l即为左边界

nums[m] <= target    => 右边界 * m = l + r + 1 >> 1 因为下界问题会造成死循环 注意细节

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0) return new int[]{-1, -1};
        int[] res = new int[2];
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int m = l + r >> 1;
            if (nums[m] >= target) r = m;
            else l = m + 1;
        }
        if (nums[l] != target) return new int[]{-1, -1};
        else {
            res[0] = l;
            l = 0;
            r = nums.length - 1;
            while (l < r) {
                int m = l + r + 1 >> 1;
                if (nums[m] <= target) l = m;
                else r = m - 1; 
            }
            res[1] = l;
        }
        return res;

    }
}
```

## next greater element i

`暴力` o(n^2)

```java
class Solution {
    public int[] findFirstLargerElement(int[] numA) {
        int[] numB = new int[numA.length];
        for (int i = 0; i < numA.length; i++) {
            for (int j = i + 1; j < numA.length; j++) {
                if (numA[j] > numA[i]) {
                    numB[i] = numA[j];
                    break;
                }
            }
            if (numB[i] == 0) numB[i] = -1;
        }
    }
    return numB;
}
```

重点介绍`单调栈`这种数据结构 他的用途并不广泛 主要解决 `Next Greater Element` 这种问题

`单调栈解决下一个更大元素 模板`

+ for循环要从后往前扫描元素，因为栈的结构，倒着入栈其实是正着出栈。
+ while 循环的目的是把两个“高个”元素之间的元素排除 因为他们不可能作为后续进来的元素的Next Greater Element 了
+ 时间复杂度 O(n) 虽然有两层嵌套 但是整体来看 总共有n个元素，每个元素都被push入栈了一次，而最多会被pop一次，没有任何冗余操作，所以计算规模和元素规模是相等的

```java
private static int[] nextGreaterElement(int[] nums) {
        int[] res = new int[nums.length];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums[i]) {
                stack.pop();
            }
            res[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(nums[i]);
        }
        return res;
    }
```

以上是解决更大元素值的通用模板 泛化该思路 有可能会遇到通过索引计算值的问题 此时栈中保存的就不是元素值而是元素的下标了 而遍历逻辑依然是相似的

```java
private static int[] dailyTemperatures(int[] t) {
        int[] res = new int[t.length];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = t.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && t[stack.peek()] <= t[i]) {
                stack.pop();
            }
            res[i] = stack.isEmpty() ? 0 : (stack.peek() - i);
            stack.push(i);
        }
        return res;
    }
```

循环数组的问题如何解决呢 也就是说最右边的元素它的下一个更大元素有可能存在与它的左边

+ 模拟构造 两倍长的数组 这样即使是右边的元素 也要去比较它左边的所有元素 这样才能保证这个元素是否有比他大的第一个元素
+ 同时我们不需要真的构造这样的两倍长数组 通过`取余`的操作 从而可以定位到实际数组的下标 并且覆盖掉之前可能有问题的结果（比如一开始的右边的元素栈里没有其他元素 赋值-1但是实际上左边还有更大的元素 等再遍历到它时栈里就有了 再覆盖）

```java
private static int[] nextGreaterElementsInCycle(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        //假设这个数组长度翻倍了
        for (int i = 2 * n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums[i % n]) {
                stack.pop();
            }
            res[i % n] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(nums[i % n]);
        }
        return res;
    }
```

---

原题解 `单调栈 + hash`

```java
class Solution {
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Deque<Integer> stack = new ArrayDeque<>();
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = nums2.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums2[i]) {
                stack.pop();
            }
            map.put(nums2[i], stack.isEmpty() ? -1 : stack.peek());
            stack.push(nums2[i]);
        }
        int[] res = new int[nums1.length];
        for (int i = 0; i < res.length; i++) {
            res[i] = map.get(nums1[i]);
        }
        return res;
    }
}
```

## 132 pattern

`单调栈`

思路 ：枚举 i 当存在满足条件的(j, k)时 比较k和i 如果k > i 则返回true

单调栈满足了以下条件：

+ 在while循环中判断 如果当前数字大于栈顶元素 则将栈顶元素出栈作为Math.max(k, currEle) 因为是从后往前遍历的 所以栈顶元素（也就是潜在的k的新值）的索引 一定是大于当前元素（也就是潜在的 j 的值）的索引，从而保证了(j, k) 的索引+值的关系
+ 而每次遍历i的时候 确定nums[i] 与 k的大小关系 基于第一点 已经保证了满足存在(j , k) 所以一旦当前nums[i] < k 则找到了132pattern

```java
class Solution {
    public boolean find132pattern(int[] nums) {
        Deque<Integer> stack = new ArrayDeque<>();
        int k = Integer.MIN_VALUE; 
        for (int i = nums.length - 1; i >= 0; i--) {
            if (nums[i] < k) return true;
            while (!stack.isEmpty() && stack.peek() < nums[i]) {
                k = Math.max(k, stack.pop());
            }
            stack.push(nums[i]);
        }
        return false;
    }
}
```

## nth digit

`math`

1.  确定 n 所在 数字 的 位数 ，记为 digit ；count = digit * start * 9
2.  确定 n 所在的 数字 ，记为 num ；      num = start + (n - 1) / digit
3.  确定 n 是 num 中的哪一数位，并返回结果。(n -1) % digit

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

## ugly number ii

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
            if (dp[i] == n1) a++;
            if (dp[i] == n2) b++;
            if (dp[i] == n3) c++; 
        }
        return dp[n - 1];
    }
}
```

## ugly-number

```java
class Solution {
    public boolean isUgly(int n) {
        if (n <= 0) return false;
        while ((n & 1) == 0) n >>= 1;
        while (n % 3 == 0) n /= 3;
        while (n % 5 == 0) n /= 5;
        return n == 1;
    }
}
```

## palindrome number

负数 false

计算 x的倒叙temp 返回temp == x

```java
class Solution {
    public boolean isPalindrome(int x) {
        if (x < 0) return false;
        int temp = 0, num = x;
        while (x != 0) {
            temp = temp * 10 +  x % 10;
            x /= 10;          
        }
        return temp == num; 
    }
}
```

## search insert position

`二分`

```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int m = l + r >> 1;
            if (target == nums[m]) return m;
            else if (target > nums[m]) l = m + 1;
            else r = m - 1;
        }
        return l;
    }
}
```

## implement strstr

`brute force`

```java
class Solution {
    public int strStr(String haystack, String needle) {
        int m = haystack.length(), n = needle.length();
        for (int i = 0; i <= m - n; i++) {
            int j = 0;
            for (; j < n; j++) {
                if (haystack.charAt(i + j) != needle.charAt(j)) {
                    break;
                }
            }
            if (j == n) {
                return i;
            }
        }
        return -1;
    }
}
```

## next permutation

`核心思路`

从右往左寻找 第一组满足要求的顺序对 O(N)

第一次从右向左遍历 寻找第一个满足 左边小于右边的元素

第二次从右向左边元素对应的索引遍历  寻找第一个满足左边该元素 小于当前右边的元素

swap 这两个元素

则 i + 1 到 nums.length 一定是降序 

reverse 即可

如果数组一开始就是降序数组 则对应左边下标为 -1 越界 直接进行reverse

该API已收录到Cpp next_permutation中

```java
class Solution {
    public void nextPermutation(int[] nums) {
        if (nums.length <= 1) return;
        int i = nums.length -2, j = nums.length - 1;
        while (i >= 0 && nums[i] >= nums[i + 1]) i--;
        if (i >= 0) {
            while (j >0 && nums[i] >= nums[j]) j--;
            int temp = nums[i]; nums[i] = nums[j]; nums[j] = temp;
        }
        Arrays.sort(nums, i + 1, nums.length);
    }
}
```

实现reverse的代码 代替Arrays.sort

```java
int i = i + 1, j = nums.length - 1;
whlie (i < j) {
    if (i != j) {
        nums[i] = nums[i] ^ nums[j];
        nums[j] = nums[i] ^ nums[j];
        nums[i] = nums[i] ^ nums[j];
    }
    i++;j--;
}
```

## rotate image

原地置换 

顺时针旋转 = 水平翻转 + 左对角线翻转

```java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        //水平翻转
        for (int i = 0; i < (n / 2); i++) {
            for (int j = 0; j < n; j++) {
                swap(matrix, i,j, n - i - 1, j );
            }
        }
        //对角线翻转
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (i == j) continue;
                swap (matrix, i, j, j, i);
            }
        }
        
    }
    void swap(int[][] matrix, int i, int j, int p ,int q) {
        int temp = matrix[i][j];
        matrix[i][j] = matrix[p][q];
        matrix[p][q] = temp;
    }
}
```

## minimum path sum

`dp`

```java
class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for(int i = 1; i < m; i++) dp[i][0] = dp[i - 1][0] + grid[i][0];
        for (int j = 1; j < n; j++) dp[0][j] = dp[0][j - 1] +  grid[0][j];
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i -1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        
        return dp[m - 1][n - 1];
    }
}
```

`dp` 避免边界赋值

```java
class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == 1 || j == 1) {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j -1]) + grid[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1];
                }
                    
            }
            
        }
        return dp[m][n];
    }
}
```

## sort colors

刷漆思想

两个指针分别代表0 和 1 的右边界

遍历数组 拿到元素后先置为2 然后判断该元素

更新指针边界并赋正确值

```java
class Solution {
    public void sortColors(int[] nums) {
        int l = 0, r = 0;
        for (int i = 0; i < nums.length; i++) {
            int temp = nums[i];
            nums[i] = 2;
            if (temp < 2) nums[r++] = 1;
            if (temp < 1) nums[l++] = 0;
        }
    }
}
```

## minimum window 

`滑动窗口`

+ 注意先修改need[idx] 再修改索引

+ 过滤左边界多余包含元素时 count不需要变化

```java
public class LeetCode_76 {
    public static String minWindow(String s, String t) {
        //首先创建的是need数组表示每个字符在t中需要的数量，用ASCII码来保存
        //加入need[76] = 2，表明ASCII码为76的这个字符在目标字符串t中需要两个，如果是负数表明当前字符串在窗口中是多余的，需要过滤掉
        int[] need = new int[128];
        //按照字符串t的内容向need中添加元素
        for (int i = 0; i < t.length(); i++) {
            need[t.charAt(i)]++;
        }
        /*
        l: 滑动窗口左边界
        r: 滑动窗口右边界
        size: 窗口的长度
        count: 当次遍历中还需要几个字符才能够满足包含t中所有字符的条件，最大也就是t的长度
        start: 如果有效更新滑动窗口，记录这个窗口的起始位置，方便后续找子串用
         */
        int l = 0, r = 0, size = Integer.MAX_VALUE, count = t.length(), start = 0;
        //循环条件右边界不超过s的长度
        while (r < s.length()) {
            char c = s.charAt(r);
            //表示t中包含当前遍历到的这个c字符，更新目前所需要的count数大小，应该减少一个
            if (need[c] > 0) {
                count--;
            }
            //无论这个字符是否包含在t中，need[]数组中对应那个字符的计数都减少1，利用正负区分这个字符是多余的还是有用的
            need[c]--;
            //count==0说明当前的窗口已经满足了包含t所需所有字符的条件
            if (count == 0) {
                //如果左边界这个字符对应的值在need[]数组中小于0，说明他是一个多余元素，不包含在t内
                while (l < r && need[s.charAt(l)] < 0) {
                    //在need[]数组中维护更新这个值，增加1
                    need[s.charAt(l)]++;
                    //左边界向右移，过滤掉这个元素
                    l++;
                }
                //如果当前的这个窗口值比之前维护的窗口值更小，需要进行更新
                if (r - l + 1 < size) {
                    //更新窗口值
                    size = r - l + 1;
                    //更新窗口起始位置，方便之后找到这个位置返回结果
                    start = l;
                }
                //先将l位置的字符计数重新加1
                need[s.charAt(l)]++;
                //重新维护左边界值和当前所需字符的值count
                l++;
                count++;
            }
            //右移边界，开始下一次循环
            r++;
        }
        return size == Integer.MAX_VALUE ? "" : s.substring(start, start + size);
    }
}
```

## word search

```java
public class Solution {
public boolean exist(char[][] board, String word) {
    for(int i = 0; i < board.length; i++)
        for(int j = 0; j < board[0].length; j++){
            if(exist(board, i, j, word, 0))
                return true;
        }
    return false;
}
private boolean exist(char[][] board, int i, int j, String word, int ind){
    if(ind == word.length()) return true;
    if(i > board.length-1 || i <0 || j<0 || j >board[0].length-1 || board[i][j]!=word.charAt(ind))
        return false;
    board[i][j]='*';
    boolean result =     exist(board, i-1, j, word, ind+1) ||
                        exist(board, i, j-1, word, ind+1) ||
                        exist(board, i, j+1, word, ind+1) ||
                        exist(board, i+1, j, word, ind+1);
    board[i][j] = word.charAt(ind);
    return result;
}
```

## unique binary search trees

`dp`

标签：动态规划
假设 n 个节点存在二叉排序树的个数是 G (n)，令 f(i) 为以 i 为根的二叉搜索树的个数，则

> G(n) = f(1) + f(2) + f(3) + f(4) + ... + f(n)

当 i 为根节点时，其左子树节点个数为 i-1 个，右子树节点为 n-i，则

> f(i) = G(i-1)*G(n-i)

综合两个公式可以得到 卡特兰数 公式

> G(n) = G(0)*G(n-1)+ G(1) *(n-2)+...+G(n-1) * G(0)

```java
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        
        for(int i = 2; i < n + 1; i++)
            for(int j = 1; j < i + 1; j++) 
                dp[i] += dp[j-1] * dp[i-j];
        
        return dp[n];
    }
}
```

## flatten binary tree to linked list

`post order traversal from (right, left, root) order`

如果按照先序遍历的顺序依次设指针指向的话由于左 根 右的遍历顺序的关系 右子树会造成丢失

所以可以按照后序遍历的变形 右 左 根  每次让当前根节点指向上一个节点（使用全局变量prev 保存）

由于在遍历根节点时左右节点都已经遍历过了 所以不会出现丢失的问题

```java
class Solution {
    TreeNode prev = null;
    public void flatten(TreeNode root) {
        if (root == null) return;
        flatten(root.right);
        flatten(root.left);
        root.right = prev;
        root.left = null;
        prev = root;
    }
}
```

## binary tree maximum path sum

`recursion`

```java
class Solution {
    int result = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        dfs(root);
        return result;
    }

    // 函数功能：返回当前节点能为父亲提供的贡献，需要结合上面的图来看！
    private int dfs(TreeNode root) {
        // 如果当前节点为叶子节点，那么对父亲贡献为 0
        if(root == null) return 0;
        // 如果不是叶子节点，计算当前节点的左右孩子对自身的贡献left和right
        int left = dfs(root.left);
        int right = dfs(root.right);
        // 更新最大值，就是当前节点的val 加上左右节点的贡献。
        result = Math.max(result, root.val + left + right);
        // 计算当前节点能为父亲提供的最大贡献，必须是把 val 加上！
        int max = Math.max(root.val + left, root.val + right);
        // 如果贡献小于0的话，直接返回0即可！
        return max < 0 ? 0 : max;
    }
}
```

## longest consecutive sequence



```java
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) set.add(num);
        int res = 0;
        for (int num : nums) {
            //跳过连续序列的中间元素 保证是从序列的最左端开始统计当前序列最大长度
            if (!set.contains(num - 1)) {
                int head = num;
                int curr = 1;
                while (set.contains(head + 1)) {
                    curr ++;
                    head ++;
                }
                res = Math.max(res, curr);
            }
        }
        return res;
    }
}
```

## single number

`位运算`

> 0 ^ x = x

> x ^ x = 0

数组各元素相异或 结果为只出现一次的数字

+ 如果有两个只出现一次的数字 同理 将该数组分为两个分别包含单个数的数组 后同理

```java
class Solution {
    public int singleNumber(int[] nums) {
        int x = 0;
        for (int num : nums)  x ^= num;
        return x;
    }
}
```

## word break

`dfs + 记忆化搜索`

![image.png](https://pic.leetcode-cn.com/78fd09b2deabeae972809c2795ddb8be96720b8e62377cf01b7f70e7fb3dbf8c-image.png)

如果没有记忆化搜索会产生大量重复计算 可以利用一个数组来保存中间结果

数组的索引代表从0到idx 的字串能否匹配 在之前判断如果不匹配 还相当于剪枝的操作

```java
class Solution {
    int len;
    int[] memory;
    public boolean wordBreak(String s, List<String> wordDict) {
        len = s.length();
        memory = new int[len + 1];
        return recur (0, s, wordDict);
    }
    boolean recur (int start, String s, List<String> wordDict) {
        if (start == len) {
            return true;
        }
        if (memory[start] == 1) return true;
        if (memory[start] == -1) return false;
        
        for (int i = start + 1; i <= len; i++) {
            String prefix = s.substring(start, i);
            if (wordDict.contains(prefix) && recur (i, s, wordDict)) {
                memory[start] = 1;
                return true;
            }
        }
        memory[start] = -1;
        return false;
    }
    
}
```

> 也可以用BFS 或者动态规划来解决https://leetcode-cn.com/problems/word-break/solution/shou-hui-tu-jie-san-chong-fang-fa-dfs-bfs-dong-tai/

## course schedule

`DAG`有向无环图

`dfs`https://leetcode-cn.com/problems/course-schedule/solution/course-schedule-tuo-bu-pai-xu-bfsdfsliang-chong-fa/

```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for(int i = 0; i < numCourses; i++)
            adjacency.add(new ArrayList<>());
        int[] flags = new int[numCourses];
        for(int[] cp : prerequisites)
            adjacency.get(cp[1]).add(cp[0]);
        for(int i = 0; i < numCourses; i++)
            if(!dfs(adjacency, flags, i)) return false;
        return true;
    }
    private boolean dfs(List<List<Integer>> adjacency, int[] flags, int i) {
        if(flags[i] == 1) return false;
        if(flags[i] == -1) return true;
        flags[i] = 1;
        for(Integer j : adjacency.get(i))
            if(!dfs(adjacency, flags, j)) return false;
        flags[i] = -1;
        return true;
    }
}
```

## kth smallest element int a bst

`recursion` 中序遍历 二叉搜索树有序

```java
class Solution {
    int k;
    int res;
    public int kthSmallest(TreeNode root, int k) {
        this.k = k;
        dfs (root);
        return res;
    }
    private void dfs (TreeNode node) {
        if (node == null) return;
        dfs (node.left);
        if (--k == 0) {
            res = node.val;
            return;
        }
        dfs (node.right);
    }
}
```

`iteration`

其实这也是中序遍历 迭代模板  都是要借用 栈  作为数据结构

```java
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        while(root != null || !queue.isEmpty()) {
            while (root != null) {
                queue.push(root);
                root = root.left;
            }
            root = queue.pop();
            if (--k == 0) return root.val;
            root = root.right;
        }
        return -1;
    }
}
```

先序遍历 迭代模板

```java
public List<Integer> preorderTraversal(TreeNode root) {
		if (root == null) {
			return null;
		}
		List<Integer> list = new ArrayList<Integer>();

		Stack<TreeNode> s = new Stack<TreeNode>();
		s.push(root);

		while (!s.isEmpty()) {
			
			TreeNode node = s.pop();
			list.add(node.val);
			
			if (node.right != null) {
				s.push(node.right);
			}
			
			if (node.left != null) {
				s.push(node.left);
			}
		}
		
		return list;
	}
```

后序遍历 迭代模板

```java
public static List<Integer> postorderTraversal(TreeNode root) {
		if (root == null) {
			return null;
		}
		List<Integer> list = new ArrayList<Integer>();

		Stack<TreeNode> s = new Stack<TreeNode>();
		
		s.push(root);
		
		while( !s.isEmpty() ) {
			TreeNode node = s.pop();
			if(node.left != null) {
				s.push(node.left);
			}
			
			if(node.right != null) {
				s.push(node.right);
			}
			
			list.add(0, node.val);
		}
		
		return list;
	}

```

## path sum

`recursion`

```java
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        if (root.left == null && root.right == null) return root.val == targetSum;
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum -  root.val);
    }
    
}
```



## kth largest element in an array

`priority queue`

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> heap = new PriorityQueue<>();
        for (int num : nums) {
            heap.add(num);
            if (heap.size() > k) {
                heap.poll();
            }
        }
        return heap.peek();
    }
}
```

`recursive quicksort`

```java
public class Solution {

    public int findKthLargest(int[] nums, int k) {
        int len = nums.length;
        int left = 0;
        int right = len - 1;

        // 转换一下，第 k 大元素的下标是 len - k
        int target = len - k;

        while (true) {
            int index = partition(nums, left, right);
            if (index == target) {
                return nums[index];
            } else if (index < target) {
                left = index + 1;
            } else {
                right = index - 1;
            }
        }
    }
 int partition(int[] a, int begin, int end) {
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
}
```

## maximal square

`dp`

![image-20210825100220988](C:\Users\64580\AppData\Roaming\Typora\typora-user-images\image-20210825100220988.png)

```java
// 伪代码
if (grid[i - 1][j - 1] == '1') {
    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1;
}
```

+ 木桶短板效应 取左 上 左上三个 以右下角构成最大正方形的面积的最小值
+ 扩充 左上边界 赋值为0 作为边界判断 简化代码

```java
public int maximalSquare(char[][] matrix) {
    // base condition
    if (matrix == null || matrix.length < 1 || matrix[0].length < 1) return 0;

    int height = matrix.length;
    int width = matrix[0].length;
    int maxSide = 0;

    // 相当于已经预处理新增第一行、第一列均为0
    int[][] dp = new int[height + 1][width + 1];

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (matrix[row][col] == '1') {
                dp[row + 1][col + 1] = Math.min(Math.min(dp[row + 1][col], dp[row][col + 1]), dp[row][col]) + 1;
                maxSide = Math.max(maxSide, dp[row + 1][col + 1]);
            }
        }
    }
    return maxSide * maxSide;
}
```

## top-k-frequent-elements

O(N) bucket sort

```java
 public static int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer>[] bucket = new List[nums.length + 1];
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (int key : map.keySet()) {
            int frequency = map.get(key);
            if (bucket[frequency] == null) {
                bucket[frequency] = new ArrayList<>();
            }
            bucket[frequency].add(key);
        }
        List<Integer> list = new ArrayList<>();
        for (int pos = nums.length; pos >= 0 && list.size() < k; pos --) {
            if (bucket[pos] != null) {
                list.addAll(bucket[pos]);
            }
        }
        int[] res = new int[k];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
       return res;
    }
```

## permutaion in string

`滑动窗口`

```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        if (n > m) {
            return false;
        }
        int[] cnt1 = new int[26];
        int[] cnt2 = new int[26];
        for (int i = 0; i < n; ++i) {
            ++cnt1[s1.charAt(i) - 'a'];
            ++cnt2[s2.charAt(i) - 'a'];
        }
        if (Arrays.equals(cnt1, cnt2)) {
            return true;
        }
        for (int i = n; i < m; ++i) {
            ++cnt2[s2.charAt(i) - 'a'];
            --cnt2[s2.charAt(i - n) - 'a'];
            if (Arrays.equals(cnt1, cnt2)) {
                return true;
            }
        }
        return false;
    }
}
```

`double pointer`

```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        if (n > m) {
            return false;
        }
        int[] cnt = new int[26];
        for (int i = 0; i < n; ++i) {
            --cnt[s1.charAt(i) - 'a'];
        }
        int left = 0;
        for (int right = 0; right < m; ++right) {
            int x = s2.charAt(right) - 'a';
            ++cnt[x];
            while (cnt[x] > 0) {
                --cnt[s2.charAt(left) - 'a'];
                ++left;
            }
            if (right - left + 1 == n) {
                return true;
            }
        }
        return false;
    }
}
```

## merge two binary trees

`dfs`

```java
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return root1 == null ? root2 : root1;
        }
        return merge(root1, root2);
    }
    TreeNode merge(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return root1 == null ? root2 : root1;
        }
        root1.val += root2.val;
        root1.left = merge(root1.left, root2.left);
        root1.right = merge(root1.right, root2.right);
        return root1;
    }
}
```

`bfs`

```java
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return root1 == null ? root2 : root1;
        }
        Deque<TreeNode> queue = new LinkedList<>();
        queue.add(root1);
        queue.add(root2);
        while (!queue.isEmpty()) {
            TreeNode r1 = queue.poll();
            TreeNode r2 = queue.poll();
            r1.val += r2.val;
            if (r1.left != null && r2.left != null) {
                queue.add(r1.left);
                queue.add(r2.left);
            } else if (r1.left == null) {
                r1.left = r2.left;
            }
            if (r1.right != null && r2.right != null) {
                queue.add(r1.right);
                queue.add(r2.right);
            } else if (r1.right == null) {
                r1.right = r2.right;
            }
        }
        return root1;
    }
}
```

## populating next right pointers in each node

`dfs`

```java
class Solution {
	public Node connect(Node root) {
		dfs(root);
		return root;
	}
	
	void dfs(Node root) {
		if(root==null) {
			return;
		}
		Node left = root.left;
		Node right = root.right;
		//配合动画演示理解这段，以root为起点，将整个纵深这段串联起来
		while(left!=null) {
			left.next = right;
			left = left.right;
			right = right.left;
		}
		//递归的调用左右节点，完成同样的纵深串联
		dfs(root.left);
		dfs(root.right);
	}
}

```

`bfs by queue`

```java
class Solution {
    public Node connect(Node root) {
        if (root == null) return root;
        LinkedList<Node> queue = new LinkedList<>();  //使用linkedList的api get方法 可以获取指定索引的节点
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            Node temp = queue.get(0);
            //对当前层从左往右依次进行串联
            for (int i = 1; i < size; i++) {
                temp.next = queue.get(i);
                temp = queue.get(i);
            }
            //对下一层的节点进行依次的添加
            for (int i = 0; i < size; i++) {
                temp = queue.poll();
                if(temp.left != null) queue.add(temp.left);
                if (temp.right != null) queue.add(temp.right);
            }
        }
        return root;
    }
}
```

## longest happy prefix

**本质 就是求字符串的最长公共前后缀 =》 KMP算法的核心**

KMP算法分为两步

+ 获得LPS数组 即从 0 - (i - 1) 长度i组成的字串的最长公共前后缀 数组
+ 进行匹配 遇到不match的指针下标根据LPS数组进行移动 (效率高的原因 相比与BF 省去了指针回溯的操作 利用了已匹配的字符串信息)

`求LPS数组 即本题的求最长公共前后缀` 即 `pat.subtring(0, lps[M - 1])`

```java
int[] computeLPSArray(String pat) {
    int M = pat.length();
    int len = 0; //length of previous longest prefix suffix
    int i = 1; //lps[0] is always 0
    int[] lps = new int[M];
    //the loop calculates lps[i] for i = 1 to M - 1
    while (i < M) {
        if (pat.charAt(i) == pat.charAt(len)) {
            lps[i++] = ++len;
        } else {
            //This is tricky. Consider the example
            //AAACAAAA and i = 7. The idea is similar to search step
            if (len != 0) len = lps[len - 1];
            else lps[i++] = len;
        }
    }
    return lps;
}
```

`KMP search`

```java
void KMPSearch(String txt, String pat) {
    int m = txt.length(), n = pat.length();
    if (n > m) return;
    int[] lps = computeLPSArray(pat); //next array
    int i = 0, j = 0; //idx for txt and pat
    while (i < m) {
        if (txt.charAt(i) == pat.charAt(j)) {
            i++;j++;
        }
        if (j == n) {
            System.out.println("Found pattern at index " + (i - j));
            j = lps[j - 1];
        } else if(i < m && txt.charAt(i) != pat.charAt(j)) {
            if (j != 0) j = lps[j - 1];
            else i = i + 1;
        }
    }
}
```

