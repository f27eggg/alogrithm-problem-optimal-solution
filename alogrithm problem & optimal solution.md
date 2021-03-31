

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
    public ListNode reversList(ListNode head) {
		if (head == null || head.next == null) return head;
        ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
    	return newHead;
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

`fast & slow pointer` O(n) O(1)

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

## remove nth  node from end of list

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
    public removeDuplicates(int[] nums) {
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

`extra space` O(n) O(n)

```java
class Solution {
	public void retate (int[] nums, int k) {
        int[] temp = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            temp[(i + k) % nums.length] = nums[i];
        }
        System.Arraycopy(temp, 0, nums, 0, temp.length);
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

`ArrayDeque` O(n) time

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
            dq.offer(i);
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



## Lowest Common Ancestor of Deepest Leaves

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

## permutations II

`backtracking`



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

## permutation

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

## permutation II

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

`O(1) time O(n) space` tricky solution

```java
public int major(int[] nums)) {
    int major = nums[0], count = 1;
    for (int i = 1; i < nums.length; i++) {
        if (count == 0) {
            major = nums[i];
            count = 1;
        } else if (major == nums[i]) {
            count ++;
        } else count--;
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

## find the largest value om each tree row

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
    dfs(grid, r, c + 1);
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

`二分法`  细节较多 基于模板要注意细节 这里返回的right 而不是mid

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
        if (nums == null || nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        int res = Math.max(dp[0], dp[1]);
        for (int i = 2; i < nums.length; i++) {
			dp[i] = Math.max(dp[i-1],dp[i-2] + nums[i]);
            res = Math.max(dp[i],res);
        }
        return res;
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
    public int maxProfit(int[] prices) {
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

`optima dp`

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



## implement trie prefix tree

```java
class Trie {
    class TrieNode {

        private boolean isEnd;
        TrieNode[] next;

        public TrieNode() {
            isEnd = false;
            next = new TrieNode[26];
        }
    }
    private TrieNode root;

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
            node = node.next[ch - 'a']; 
            if (node == null) {
                return false;
            } 
        }
        return node.isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char ch : prefix.toCharArray()) {
            node = node.next[ch - 'a'];
            if (node == null) {
                return false;
            }
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

## generate parenthese

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

## power of two 

`2的幂二进制下汉明重量为1 `

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

## LRU cache lcci

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

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] cost = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) cost[i][0] = i;
        for (int i = 1; i <= n; i++) cost[0][i] = i;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (word1.charAt(i) == word2.charAt(j)) cost[i + 1][j + 1] = cost[i][j];
                else {
                    int a = cost[i][j];
                    int b = cost[i + 1][j];
                    int c = cost[i][j + 1];
                    cost[i + 1][j + 1] = a < b ? (a < c ? a : c) : (b < c ? b : c);
                    cost[i + 1][j + 1] ++;
                }
            }
        }
        return cost[m][n];
    }
}
```

## longest increasing subsequence

## decode ways

## longest valid parentheses

## maximal rectangle

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
                if (p.charAt(j - 1) == '*') {
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

