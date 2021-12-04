import java.util.*;

/**
 * 链表相关题目
 *
 */
public class Linked_List {

    /**
     * 反转链表 - https://leetcode-cn.com/problems/reverse-linked-list/
     * @param head 反转前的链表头
     * @return 反转后的链表头
     */
    public ListNode ReverseList(ListNode head) {
        /*迭代写法*/
        /*ListNode tmp, newHead = null;
        while (head!=null) {
            tmp = head.next;
            //插一个节点到newHead前
            head.next = newHead;
            //修改newHead新的头节点
            newHead = head;
            head = tmp;
        }
        return newHead;*/

        /*递归写法*/
        if (head.next == null) return head;
        ListNode newHead = ReverseList(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    /**
     * 反转链表的部分区间 - https://leetcode-cn.com/problems/reverse-linked-list-ii/
     * @param head 反转前链表头
     * @param m 待反转开始区间
     * @param n 待反转结束区间
     * @return 反转后链表头
     */
    public ListNode ReverseListBetween(ListNode head, int m, int n) {
        /*迭代写法*/
        /*
        //设置一个傀儡节点
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode precessor = dummy;
        for (int i = 1; i < m; i++) precessor = precessor.next;

        //头插法
        head = precessor.next;
        for(int j = m; j < n; j++){
            ListNode nextNode = head.next;
            head.next = nextNode.next;
            nextNode.next = precessor.next;
            precessor.next = nextNode;
        }

        return dummy.next;
        */

        /*递归写法*/
        if (m == 1) {
            return ReverseListN(head, n);
        }
        // 前进到反转的起点触发 base case
        head.next = ReverseListBetween(head.next, m - 1, n - 1);
        return head;
    }

    ListNode successor = null; // 后驱节点
    /**
     * 反转以 head 为起点的 n 个节点，返回新的头结点
     * @param head 反转前链表头
     * @param n 待反转的前n部分,n小于等于链表长度
     * @return 反转后新的链表头
     */
    public ListNode ReverseListN(ListNode head, int n) {
        if (n == 1) {
            // 记录第 n + 1 个节点
            successor = head.next;
            return head;
        }
        // 以 head.next 为起点，需要反转前 n - 1 个节点
        ListNode last = ReverseListN(head.next, n - 1);

        head.next.next = head;
        // 让反转之后的 head 节点和后面的节点连起来
        head.next = successor;
        return last;
    }

    /**
     * K个一组翻转链表 - https://leetcode-cn.com/problems/reverse-nodes-in-k-group/
     * @param head 翻转前的链表头
     * @param k k个一组
     * @return 反转后的链表头
     */
    public ListNode reverseListKGroup(ListNode head, int k) {
        if (head == null) return null;
        // 区间 [start, end) 包含 k 个待反转元素
        ListNode start,end;
        start = end = head;
        for (int i = 0; i < k; i++) {
            // 如果不足k个,不需要反转
            if (end == null) return head;
            end = end.next;
        }
        // 反转前K个元素
        ListNode newHead = reverseList(start,end);
        // 递归反转后续链表并连接起来
        start.next = reverseListKGroup(end,k);
        return newHead;
    }

    /**
     * 反转区间 [start, end) 的元素，注意是左闭右开
     * @param start 区间开始
     * @param end 区间结束,注意不包括此
     * @return 反转后的链表头
     */
    public ListNode reverseList(ListNode start,ListNode end){
        ListNode current, next, newHead = null;
        current = start;
        //头插法
        while (current != end) {
            next = current.next;
            current.next = newHead;
            newHead = current;
            current = next;
        }
        return newHead;
    }

    /**
     * 旋转链表
     * https://leetcode-cn.com/problems/rotate-list/
     * 给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null) return head;

        //计算出链表的长度
        int n = 0;
        ListNode tmp = head;
        while (tmp!=null) {
            tmp = tmp.next;
            n++;
        }

        //旋转一圈又回到起点
        if (k%n == 0) return head;

        //找到新链表尾
        int newTailPos = n - k%n;
        tmp = head;
        while (--newTailPos > 0){
            tmp = tmp.next;
        }

        //找到新链表头
        ListNode newHead = tmp.next;
        tmp.next = null;

        //拼接新链表尾和旧链表头
        tmp = newHead;
        while (tmp.next!=null) {
            tmp = tmp.next;
        }
        tmp.next = head;

        return newHead;
    }

    /**
     * 合并两个有序链表
     * https://leetcode-cn.com/problems/merge-two-sorted-lists/
     * l1 和 l2 的节点数目范围是 [0, 50]
     * l1 和 l2 均按 非递减顺序 排列
     * @param l1 -100 <= Node.val <= 100
     * @param l2 -100 <= Node.val <= 100
     * @return 将两个升序链表合并为一个新的升序链表并返回。
     *          新链表是通过拼接给定的两个链表的所有节点组成的。
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;

        //如果 l1的初始节点值>l2的初始节点值 两者交换一下位置
        if (l1.val > l2.val) {
            ListNode tmp = l2;
            l2 = l1;
            l1 = tmp;
        }
        ListNode newHead = l1;
        while (l2 != null) {
            //如果l1用完,拼接l2即可结束
            if (l1.next == null) {
                l1.next = l2;
                break;
            }

            //如果l1没用完,并且l2的值在[l1.val,l1.next.val]之间
            if (l2.val >= l1.val && l2.val <= l1.next.val){
                ListNode l1_next = l1.next;
                l1.next = l2;
                l2 = l2.next;
                l1.next.next = l1_next;
                l1 = l1.next;
            }
            //如果l1没用完,并且l2的值不能放在当前[l1.val,l1.next.val]之间
            else l1 = l1.next;
        }

        return newHead;
    }

    /**
     * 合并K个有序链表
     * 给你一个链表数组，每个链表都已经按升序排列。
     * @param lists k == lists.length && 0 <= k <= 10^4
     *              && 0 <= lists[i].length <= 500
     *              && -10^4 <= lists[i][j] <= 10^4
     *              && lists[i] 按 升序 排列
     *              && lists[i].length 的总和不超过 10^4
     * @return 请你将所有链表合并到一个升序链表中，返回合并后的链表。
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;

        //设置傀儡节点,并指向第一条链表
        ListNode dummy = new ListNode(-1);

        /*
        dummy.next = lists[0];
        //从第二条链表开始进行合并
        for (int i = 1; i < lists.length; i++) {
            ListNode cur = dummy;

            ListNode l1 = cur.next;
            ListNode l2 = lists[i];
            while (l1!=null && l2!=null) {
                if (l1.val < l2.val) {
                    cur.next = l1;
                    l1 = l1.next;
                } else {
                    cur.next = l2;
                    l2 = l2.next;
                }
                cur = cur.next;
            }

            cur.next = l1 == null ? l2 : l1;
        }*/

        /*利用优先队列最小堆的特点*/
        Queue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });
        for (ListNode list : lists) {
            if (list!=null) pq.add(list);
        }
        ListNode cur = dummy;
        while (!pq.isEmpty()) {
            ListNode nextNode = pq.poll();
            cur.next = nextNode;
            cur = cur.next;
            if (nextNode.next != null) {
                pq.add(nextNode.next);
            }
        }

        return dummy.next;
    }

    /**
     * @see DailyQuestion#oddEvenList(ListNode)
     * 把所有的奇数节点编号和偶数节点编号分别排在一起
     * 时间复杂度O(n)
     * 空间复杂度O(1)
     */

    /**
     * @see DailyQuestion#insertionSortList(ListNode)
     * 插入排序对链表升序排序
     * 时间复杂度O(n^2)
     * 空间复杂度O(1)
     */

    /**
     * @see DailyQuestion#sortList(ListNode)
     * 归并排序对链表升序排序
     * 时间复杂度O(nlogn)
     * 空间复杂度O(1)
     */
}

/**
 * 链表数据结构
 */
class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

/**
 * 二叉树数据结构
 */
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int val) { this.val = val; }
}