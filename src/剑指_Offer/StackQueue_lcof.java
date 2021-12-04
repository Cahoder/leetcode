package 剑指_Offer;

import java.util.*;

/**
 * 栈&队列类题目
 */
public class StackQueue_lcof {


    /**
     * 剑指 Offer 09. 用两个栈实现队列 - (简单)
     * https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/
     * 用两个栈实现一个队列。
     * 队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead，
     * 分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead操作返回 -1 )
     * 提示: 1 <= values <= 10000 最多会对 appendTail、deleteHead 进行 10000 次调用
     */
    static class CQueue {

        public CQueue() {

        }

        /*Stack<Integer> stack_in = new Stack<>();
        Stack<Integer> stack_out = new Stack<>();

        public void appendTail(int value) {
            stack_in.push(value);
        }

        public int deleteHead() {
            if (stack_in.isEmpty()) return -1;

            while (!stack_in.isEmpty()) {
                stack_out.push(stack_in.pop());
            }

            int head = stack_out.pop();

            while (!stack_out.isEmpty()) {
                stack_in.push(stack_out.pop());
            }

            return head;
        }*/

        /*Queue<Integer> stack_in = new LinkedList<>();
        Queue<Integer> stack_out = new LinkedList<>();

        public void appendTail(int value) {
            while (!stack_out.isEmpty()) {
                stack_in.offer(stack_out.poll());
            }

            stack_in.offer(value);
        }

        @SuppressWarnings("all")
        public int deleteHead() {
            if (stack_in.isEmpty() && stack_out.isEmpty()) return -1;

            while (!stack_in.isEmpty()) {
                stack_out.offer(stack_in.poll());
            }

            return stack_out.poll();
        }*/

        LinkedList<Integer> stack_in = new LinkedList<>();
        LinkedList<Integer> stack_out = new LinkedList<>();

        public void appendTail(int value) {
            stack_in.addLast(value);
        }
        public int deleteHead() {
            if (stack_out.isEmpty()) {
                if (stack_in.isEmpty()) return -1;
                else {
                    while (!stack_in.isEmpty()) {
                        stack_out.addLast(stack_in.removeLast());
                    }
                }
            }
            return stack_out.removeLast();
        }

    }

    /**
     * 剑指 Offer 30. 包含min函数的栈 - (简单)
     * https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/
     * 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，
     * 调用 min 及 push 及 pop 的时间复杂度都是 O(1)。
     * 提示: 各函数的调用总次数不超过 20000 次
     */
    static class MinStack {

        /** initialize your data structure here. */
        public MinStack() {

        }

        /*//方法一: 维护一个保存最小值的栈
        Stack<Integer> stack = new Stack<>();
        Stack<Integer> min_stack = new Stack<>();

        public void push(int x) {
            stack.push(x);
            //栈顶保存当前为止最小值
            if (min_stack.isEmpty() || min_stack.peek() >= x) min_stack.push(x);
        }

        public void pop() {
            //如果当前出栈元素为最小值,那么需要更新最小值
            if (stack.pop().equals(min())) min_stack.pop();
        }

        public int top() {
            return stack.peek();
        }

        public int min() {
            return min_stack.peek();
        }*/

        /*//方法二: 维护当前阶段最小值的变量
        Stack<Integer> stack = new Stack<>();
        int min = Integer.MAX_VALUE;

        public void push(int x) {
            if (x <= min) {
                //如果出现了更小的值,先保存住当前阶段最小值
                stack.push(min);
                min = x;
            }
            stack.push(x);
        }

        public void pop() {
            //更新当前阶段最小值
            if (stack.pop().equals(min))
                min = stack.pop();
        }

        public int top() {
            return stack.peek();
        }

        public int min() {
            return min;
        }*/

        //方法三: 使用辅助链表类
        private class Node {
            int min;
            int val;
            Node next;
            public Node(int min, int val, Node next) {
                this.min = min;
                this.val = val;
                this.next = next;
            }
        }
        private Node head;

        public void push(int x) {
            if (head == null) head = new Node(x,x,null);
            else head = new Node(Math.min(x,head.min),x,head);
        }

        public void pop() {
            head = head.next;
        }

        public int top() {
            return head.val;
        }

        public int min() {
            return head.min;
        }

    }

    /**
     * 剑指 Offer 31. 栈的压入、弹出序列 - (中等)
     * https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/
     * @param pushed 栈的压入顺序 0 <= pushed.length == popped.length <= 1000 && pushed 是 popped 的排列。
     * @param popped 栈的弹出顺序 0 <= pushed[i], popped[i] < 1000 && popped 是 pushed 的排列。
     * @return 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
     *         注意: 假设压入栈的所有数字均不相等。
     */
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        //思路1: 直接在数组上模拟栈操作, 时间O(n^2) 空间O(1)
        /*int push_idx = 0, popped_idx = 0;
        while (push_idx < pushed.length && popped_idx < popped.length) {
            if (pushed[push_idx] == -1) {
                //首先需要跳过入栈序列中已使用的
                push_idx++;
                continue;
            }

            if (pushed[push_idx] != popped[popped_idx]) {
                //表示入栈
                push_idx++;
            }
            else {
                //表示出栈
                popped_idx++;
                pushed[push_idx] = -1;  //标识为已使用
                while (push_idx > 0 && pushed[push_idx] == -1) {
                    push_idx--;
                }
            }
        }
        return push_idx == 0 && popped_idx == popped.length;*/

        //思路2: 定义一个新的栈进行校验, 时间O(n) 空间O(n)
        int[] stack = new int[pushed.length];
        int stack_idx = 0;
        for (int popped_idx = 0, push_idx = 0; push_idx < pushed.length;) {
            //先入栈
            stack[stack_idx++] = pushed[push_idx++];
            //然后看看能否出栈
            while (stack_idx > 0 && popped_idx < popped.length && stack[stack_idx-1] == popped[popped_idx]) {
                stack_idx--;
                popped_idx++;
            }
        }
        return stack_idx == 0;

        //思路3: 优化思路2,复用pushed中的空间, 时间O(n) 空间O(1)
        /*int i = 0, j = 0;
        for (int e : pushed) {
            pushed[i] = e;
            while (i >= 0 && pushed[i] == popped[j]) {
                j++;
                i--;
            }
            i++;
        }
        return i == 0;*/
    }

    /**
     * 剑指 Offer 58 - I. 翻转单词顺序 - (简单)
     * https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/
     * @param s 无空格字符构成一个单词。
     *          输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
     *          如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
     * @return 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。
     * 为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。
     */
    public String reverseWords(String s) {
        if (s.equals("")) return "";

        /*思路1： 双指针*/
        StringBuilder stb = new StringBuilder();
        s = s.trim(); // 删除首尾空格
        int j = s.length() - 1, i = j;
        while(i >= 0) {
            while(i >= 0 && s.charAt(i) != ' ') i--; // 搜索首个空格
            stb.append(s, i + 1, j + 1); // 添加单词
            if (i+1 != 0) stb.append(" ");
            while(i >= 0 && s.charAt(i) == ' ') i--; // 跳过单词间空格
            j = i; // j 指向下个单词的尾字符
        }

        /*思路2： 字符串分割*/
        /*String[] strings = s.trim().split(" +");
        for (int i = strings.length - 1; i >= 0; i--) {
            stb.append(strings[i]);
            if (i != 0) stb.append(" ");
        }*/

        return stb.toString();
    }

    /**
     * 剑指 Offer 59 - I. 滑动窗口的最大值
     * https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/
     * @param nums 数组
     * @param k 你可以假设 k 总是有效的，在输入数组不为空的情况下，1 ≤ k ≤ 输入数组的大小。
     * @return 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0 || k == 0) return new int[0];
        if (k == 1) return nums;
        int n = nums.length;
        int[] maxs = new int[n-k+1];
        int max_idx = 0;

        //解法1: 暴力破解 时间O((n-k+1)k)
        /*for (int i = 0; i < n-k+1; i++) {
            int max = nums[i];
            for (int j = 1; j < k; j++) {
                if (nums[i+j] > max) max = nums[i+j];
            }
            maxs[max_idx++] = max;
        }*/

        //解法2: 贪心+队列 时间稍比解法1快点
        /*Deque<Integer> queue = new LinkedList<>();
        int max = Integer.MIN_VALUE;
        for (int num : nums) {
            max = Math.max(num, max);
            queue.addLast(num);
            if (queue.size() == k) {
                maxs[max_idx++] = max;
                //如果出去的刚好就是前阶段的max,需要在剩下里找当前阶段max
                if (queue.removeFirst().equals(max)) {
                    max = queue.stream().max(Integer::compareTo).get();
                }
            }
        }*/

        //解法3: 贪心+单调队列时间O(n)  解法类似于包含min函数的栈里维护一个最小值栈
        Deque<Integer> queue = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            int t = i-k+1;
            //如果前一个阶段的max还没去掉,先干掉
            if(t > 0 && queue.getFirst().equals(nums[t-1])) queue.removeFirst();

            //需要保证队列的单调性
            while (!queue.isEmpty() && nums[i] > queue.getLast()) queue.removeLast();
            queue.addLast(nums[i]);

            //队首就是当前阶段最大值
            if (t >= 0) maxs[t] = queue.getFirst();
        }

        return maxs;
    }

}
