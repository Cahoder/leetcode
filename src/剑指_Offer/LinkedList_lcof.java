package 剑指_Offer;

/**
 * 链表类题目
 */
public class LinkedList_lcof {

    /**
     * 剑指 Offer 06. 从尾到头打印链表 - (简单)
     * https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/
     * @param head 输入一个链表的头节点 && 0 <= 链表长度 <= 10000
     * @return 从尾到头反过来返回每个节点的值（用数组返回）
     * 时间O(n) 空间O(1)
     */
    public int[] reversePrint(ListNode head) {
        if (head == null) return new int[0];

        int ListNodeNum = 1;
        ListNode cur = head;
        while (cur.next!=null) {
            cur = cur.next;
            ListNodeNum++;
        }

        int[] ret = new int[ListNodeNum];
        int idx = ret.length-1;
        while (head!=null){
            ret[idx--] = head.val;
            head = head.next;
        }

        return ret;
    }

    /**
     * 剑指 Offer 22. 链表中倒数第k个节点 - (简单)
     * https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/
     * @param head 输入一个链表
     * @param k 输出该链表中倒数第k个节点
     * @return 为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点
     * 时间O(n) 空间O(1)
     */
    public ListNode getKthFromEnd(ListNode head, int k) {
        if (head == null) return null;

        /*
        int ListNodeNum = 1;
        ListNode cur = head;
        while (cur.next!=null) {
            cur = cur.next;
            ListNodeNum++;
        }

        //倒数第k就是正数第(链表长度-k+1)个
        int KthFromBegin = ListNodeNum - k + 1;
        while (--KthFromBegin > 0 && head != null) {
            head = head.next;
        }
        return head;*/

        //快慢指针: fast先走k步,slow再走,等到fast到尾了,slow就是倒数第k个
        ListNode fast = head;
        ListNode slow = head;
        while (k-- > 0) fast = fast.next;

        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    /**
     * 剑指 Offer 24. 反转链表 - (简单)
     * https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/
     * @param head 输入一个链表的头节点 && 0 <= 节点个数 <= 5000
     * @return 反转该链表并输出反转后链表的头节点
     * 时间O(n) 空间O(1)
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;

        /*递归版*/
        /*ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;*/

        /*迭代版*/
        ListNode newHead = null, cur = head, next;
        while (cur != null) {
            next = cur.next;
            cur.next = newHead;
            newHead = cur;
            cur = next;
        }
        return newHead;
    }

    /**
     * 剑指 Offer 25. 合并两个排序的链表 - (简单)
     * https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/
     * @param l1 递增排序的链表
     * @param l2 递增排序的链表
     * @return 合并这两个递增排序的链表并使新链表中的节点仍然是递增排序的。
     * 时间O(n) 空间O(1)
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        /*迭代*/
        /*ListNode dummy = new ListNode(-1);

        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val > l2.val) {
                cur.next = l2;
                l2 = l2.next;
            }
            else {
                cur.next = l1;
                l1 = l1.next;
            }
            cur = cur.next;
        }
        if (l1 != null) cur.next = l1;
        else cur.next = l2;

        return dummy.next;*/

        /*递归*/
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next,l2);
            return l1;
        }
        else {
            l2.next = mergeTwoLists(l1,l2.next);
            return l2;
        }
    }

    /**
     * 剑指 Offer 35. 复杂链表的复制 - (中等)
     * https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/
     * 在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。
     * @param head -10000 <= Node.val <= 10000 && Node.random 为空（null）或指向链表中的节点 && 节点数目不超过 1000
     * @return 请实现 copyRandomList 函数，能够实现复制这个复杂链表。
     */
    public Node copyRandomList(Node head) {
        if (head == null) return null;

        /* 方法一: 利用哈希表 时间O(n) 空间O(n) */
        /*//map中存的是(原节点，拷贝节点)的一个映射
        Map<Node, Node> map = new HashMap<>();
        for (Node cur = head; cur != null; cur = cur.next) {
            map.put(cur, new Node(cur.val));
        }
        //将拷贝的新的节点组织成一个链表
        for (Node cur = head; cur != null; cur = cur.next) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
        }
        return map.get(head);*/

        /* 方法二: 原地算法 时间O(n) 空间O(1) */
        // 将节点复制一份: 如 1 -> 2 -> 3 -> null 复制完就是 1 -> 1（copy） -> 2 -> 2（copy） -> 3 - > 3（copy） -> null
        for (Node cur = head; cur != null; cur = cur.next.next) {
            Node copy = new Node(cur.val);
            copy.next = cur.next;
            cur.next = copy;
        }
        // 再将复制节点的random与原节点的random对应上
        for (Node cur = head; cur != null; cur = cur.next.next) {
            cur.next.random = cur.random == null ? null : cur.random.next;
        }
        // 最后将原节点与复制节点分离开
        Node copy_head = head.next;
        for (Node node = head, temp; node.next != null;) {
            temp = node.next;
            node.next = temp.next;
            node = temp;
        }
        return copy_head;
    }

    /**
     * 剑指 Offer 52. 两个链表的第一个公共节点 - (简单)
     * https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/
     * @param headA A链表表头
     * @param headB B链表表头
     * @return 输入两个链表，找出它们的第一个公共节点
     *         如果两个链表没有交点，返回 null.
     *         在返回结果后，两个链表仍须保持原有的结构。
     *         可假定整个链表结构中没有循环。
     *         程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;

        /*方法一: 通过集合set解决 时间O(n) 空间O(n) */
        /*Set<ListNode> set = new HashSet<>();
        //先把链表A的结点全部存放到集合set中
        while (headA != null) {
            set.add(headA);
            headA = headA.next;
        }
        //然后访问链表B的结点，判断集合中是否包含链表B的结点，如果包含就直接返回
        while (headB != null) {
            if (set.contains(headB))
                return headB;
            headB = headB.next;
        }
        //如果集合set不包含链表B的任何一个结点，说明他们没有交点，直接返回null
        return null;*/

        /*方法二: 拉齐链表 时间O(n+k),k为两条链表长度差 空间O(1)*/
        /*//统计链表A和链表B的长度
        int lenA = length(headA), lenB = length(headB);
        //如果节点长度不一样，节点多的先走，直到他们的长度一样为止
        while (lenA != lenB) {
            if (lenA > lenB) {
                //如果链表A长，那么链表A先走
                headA = headA.next;
                lenA--;
            } else {
                //如果链表B长，那么链表B先走
                headB = headB.next;
                lenB--;
            }
        }
        //然后开始比较，如果他俩不相等就一直往下走
        while (headA != headB) {
            headA = headA.next;
            headB = headB.next;
        }
        //走到最后，最终会有两种可能，一种是headA为空，也就是说他们俩不相交。
        // 还有一种可能就是headA不为空，也就是说headA就是他们的交点
        return headA;*/

        /*方法三: 相互追寻 时间O(n+k),k为两条链表长度差 空间O(1) */
        // 两个结点不断的去对方的轨迹中寻找对方的身影，只要二人有交集就终会相遇，哪怕我们是在null相遇
        ListNode h1 = headA, h2 = headB;
        while (h1 != h2) {
            h1 = h1 == null ? headB : h1.next;
            h2 = h2 == null ? headA : h2.next;
        }

        return h1;
    }

    /**
     * 剑指 Offer 18. 删除链表的节点 - (简单)
     * https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/
     * 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
     * 题目保证链表中节点的值互不相同
     * @param head 单向链表的头指针
     * @param val 要删除的节点值
     * @return 返回删除后的链表的头节点。
     */
    public ListNode deleteNode(ListNode head, int val) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;

        ListNode cur = dummy;
        while (cur.next != null) {
            if (cur.next.val == val) {
                cur.next = cur.next.next;
                break;
            }
            cur = cur.next;
        }

        return dummy.next;
    }

}

/**链表数据结构**/
class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}
/**复杂链表节点数据结构**/
class Node {
    public int val;
    public Node next;
    public Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }


    public Node left;
    public Node right;
    public Node() {}
    public Node(int _val,Node _left,Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
}