package 剑指_Offer;

import java.util.*;

/**
 * 二叉树类题目
 */
public class Tree_lcof {

    /**
     * 剑指 Offer 07. 重建二叉树 - (中等)
     * https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/
     * @param preorder 某二叉树的前序遍历
     * @param inorder 某二叉树的中序遍历
     * @return 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     *         注意: 0 <= 节点个数 <= 5000
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || inorder == null) return null;
        if (preorder.length == 0 || preorder.length != inorder.length) return null;

        return buildTree(preorder, inorder, 0,0, inorder.length-1);
    }
    private TreeNode buildTree(int[] preorder, int[] inorder, int preIdx, int inStart, int inEnd){
        if (inStart > inEnd) return null;
        TreeNode root = new TreeNode(preorder[preIdx]);

        //在中序中找前序的根
        int inRootIdx = inStart;
        while (inorder[inRootIdx] != root.val && inRootIdx <= inEnd) inRootIdx++;

        root.left = buildTree(preorder, inorder, preIdx+1, inStart, inRootIdx-1);
        //右子树的根位置 = 前序中左子树数量+1
        root.right = buildTree(preorder, inorder, preIdx+inRootIdx-inStart+1, inRootIdx+1, inEnd);
        return root;
    }

    /**
     * 剑指 Offer 26. 树的子结构 - (中等)
     * https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/
     * @param A 二叉树A
     * @param B 二叉树B
     * @return 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
     *         B是A的子结构， 即 A中有出现和B相同的结构和节点值。
     *         注意: 0 <= 节点个数 <= 10000
     * 时间O(mk), m为A的节点数,k为B的节点数
     */
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) return false;
        //从当前A节点开始判断是否子结构
        if (isSub(A,B)) return true;
        //从当前A节点的子节点判断是否存在子结构
        return isSubStructure(A.left,B) || isSubStructure(A.right,B);
    }
    private boolean isSub(TreeNode A, TreeNode B) {
        //终止条件: 如果B用完了,或者B没用完但A用完
        if (B == null) return true;
        if (A == null) return false;

        if (A.val != B.val) return false;
        return isSub(A.left,B.left) && isSub(A.right,B.right);
    }

    /**
     * 剑指 Offer 27. 二叉树的镜像 - (简单)
     * https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/
     * @param root 二叉树根节点
     * @return 请完成一个函数，输入一个二叉树，该函数输出它的镜像。
     *         注意: 0 <= 节点个数 <= 1000
     */
    public TreeNode mirrorTree(TreeNode root) {
        //终止条件
        if (root == null) return null;

        /*//递归版: 交换左右子树
        TreeNode tmp = root.left;
        root.left = mirrorTree(root.right);
        root.right = mirrorTree(tmp);*/

        //迭代版: 交换左右子树
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }

        return root;
    }

    /**
     * 剑指 Offer 32 - I. 从上到下打印二叉树 - (中等)
     * https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/
     * @param root 二叉树根节点
     * @return 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
     *        注意: 节点总数 <= 1000
     */
    public int[] levelOrder(TreeNode root) {
        if (root == null) return new int[0];
        List<Integer> list = new ArrayList<>();

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            list.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }

        int[] ints = new int[list.size()];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = list.get(i);
        }
        return ints;
    }

    /**
     * 剑指 Offer 32 - II. 从上到下打印二叉树 II - (简单)
     * https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/
     * @param root 二叉树根节点
     * @return 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
     *         注意: 节点总数 <= 1000
     */
    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> list = new LinkedList<>();
        if (root == null) return list;

        /*迭代版*/
        /*Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int level_size = queue.size();
            List<Integer> tmp = new LinkedList<>();
            for (int i = 0; i < level_size; i++) {
                TreeNode node = queue.poll();
                assert node != null;
                tmp.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            list.add(tmp);
        }*/

        /*递归版*/
        levelOrder2(root,list,0);

        return list;
    }
    private void levelOrder2(TreeNode root, List<List<Integer>> list, int level) {
        if (root == null) return;
        if(list.size() <= level) list.add(new LinkedList<>());
        list.get(level).add(root.val);
        levelOrder2(root.left,list,level+1);
        levelOrder2(root.right,list,level+1);
    }

    /**
     * 剑指 Offer 32 - III. 从上到下打印二叉树 III - (中等)
     * https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/
     * @param root 二叉树根节点
     * @return 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，
     *         第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
     *         注意: 节点总数 <= 1000
     */
    public List<List<Integer>> levelOrder3(TreeNode root) {
        List<List<Integer>> list = new LinkedList<>();
        if (root == null) return list;

        /*迭代版*/
        boolean flag = true;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int level_size = queue.size();
            List<Integer> tmp = new LinkedList<>();
            for (int i = 0; i < level_size; i++) {
                TreeNode node = queue.poll();
                assert node != null;
                if (flag) tmp.add(node.val); //从左往右添加
                else tmp.add(0,node.val); //从右往左添加
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            flag = !flag;
            list.add(tmp);
        }

        /*递归版*/
        /*levelOrder3(root,list,0,true);*/

        return list;
    }
    private void levelOrder3(TreeNode root, List<List<Integer>> list, int level, boolean flag) {
        if (root == null) return;
        if(list.size() <= level) list.add(new LinkedList<>());

        //从左往右添加
        if (flag) list.get(level).add(root.val);
        //从右往左添加
        else list.get(level).add(0,root.val);

        levelOrder3(root.left,list,level+1,!flag);
        levelOrder3(root.right,list,level+1,!flag);
    }

    /**
     * 剑指 Offer 33. 二叉搜索树的后序遍历序列 - (中等)
     * https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/
     * @param postorder 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。
     * @return 如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。
     *         注意: 节点总数 <= 1000
     */
    public boolean verifyPostorder(int[] postorder) {
        if (postorder == null || postorder.length < 2) return true;

        /*//解决思路一: 二叉搜索树的中序遍历是升序数组
        int[] inorder = Arrays.copyOf(postorder, postorder.length);
        Arrays.sort(inorder);
        //通过中序+后序尝试判断二叉搜索树: 时间复杂度 O(NlogN) 空间复杂度 O(N)
        return checkBSTByInPostOrder(inorder,postorder,postorder.length-1,0,inorder.length-1);*/

        //解决思路二: 利用后序遍历倒序（根 右 左） + 辅助单调栈 时间复杂度 O(N) 空间复杂度 O(N)
        //二叉搜索树的后序遍历倒序存特点 root < right > left
        Stack<Integer> stack = new Stack<>();
        int root = Integer.MAX_VALUE;
        for(int i = postorder.length - 1; i >= 0; i--) {
            if(postorder[i] > root) return false;
            while(!stack.isEmpty() && stack.peek() > postorder[i])
                root = stack.pop();
            stack.add(postorder[i]);
        }
        return true;
    }
    private boolean checkBSTByInPostOrder(int[] inorder, int[] postorder, int postIdx, int inStart, int inEnd) {
        if (inStart > inEnd) return true;

        //在中序中找后序的根
        int inIdx = inStart;
        while (inorder[inIdx] != postorder[postIdx] && inIdx <= inEnd) inIdx++;

        //计算出当前节点的左右子树各有多少个子节点
        int left_num = inIdx-inStart;
        int right_num = inEnd-inIdx;

        //从当前节点的右子树中检查是否有比当前节点值小的节点
        for (int i = 1; i <= right_num; i++) {
            if (postorder[postIdx-i] < postorder[postIdx]) return false;
        }

        //从当前节点的左子树中检查是否有比当前节点值大的节点
        for (int i = 1; i <= left_num; i++) {
            if (postorder[postIdx-right_num-i] > postorder[postIdx]) return false;
        }

        //对当前节点的左右子树递归判断
        return checkBSTByInPostOrder(inorder, postorder, postIdx-1, inIdx+1, inEnd)
                && checkBSTByInPostOrder(inorder,postorder,postIdx-right_num-1,inStart,inIdx-1);
    }

    /**
     * 剑指 Offer 34. 二叉树中和为某一值的路径 - (中等)
     * https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/
     * @param root 二叉树根节点
     * @param sum 一个整数
     * @return 输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
     *         注意: 节点总数 <= 10000
     */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> paths = new LinkedList<>();
        LinkedList<Integer> path = new LinkedList<>();
        backTracePath(paths,path,root,sum);
        return paths;
    }
    private void backTracePath(List<List<Integer>> paths, LinkedList<Integer> path, TreeNode root, int sum) {
        if (root == null) return;

        path.addLast(root.val);
        sum -= root.val;
        if (sum == 0 && root.left == null && root.right == null) {
            paths.add(new LinkedList<>(path));
            //这里不能return,否则导致结果永远多一个
        }
        backTracePath(paths, path, root.left, sum);
        backTracePath(paths, path, root.right, sum);
        path.removeLast();
    }

    /**
     * 剑指 Offer 36. 二叉搜索树与双向链表 - (中等)
     * https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/
     * @param root 二叉树根节点
     * @return 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。
     * 要求不能创建任何新的节点，只能调整树中节点指针的指向。
     */
    public Node treeToDoublyList(Node root) {
        if (root == null) return null;

        //迭代版: 进行中序遍历
        Stack<Node> stack = new Stack<>();
        Node head = null,tail = null,pre = null,cur = root;
        while (!stack.isEmpty() || cur != null) {
            if (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            else {
                cur = stack.pop();

                /*连接当前节点与前驱节点*/
                if (pre != null) {
                    cur.left = pre;
                    pre.right = cur;
                } else {
                    head = cur;
                }
                pre = cur;
                tail = cur;
                /*连接当前节点与前驱节点*/

                cur = cur.right;
            }
        }
        //最后再处理一下头尾
        assert head != null;
        head.left = tail;
        tail.right = head;

        return head;
    }

    int max_depth = 0;
    /**
     * 剑指 Offer 55 - I. 二叉树的深度 - (简单)
     * https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/
     * @param root 二叉树根节点
     * @return 输入一棵二叉树的根节点，求该树的深度。
     * 从根节点到叶节点依次经过的节点（含根|叶节点）形成树的一条路径，最长路径的长度为树的深度。
     * 注意: 节点总数 <= 10000
     */
    public int maxDepth(TreeNode root) {
        if (root == null) return max_depth;

        //递归版:自顶向下
        //return Math.max(maxDepth(root.left),maxDepth(root.right)) + 1;

        //迭代版:自顶向下
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                assert node != null;
                if (node.left!=null) queue.offer(node.left);
                if (node.right!=null) queue.offer(node.right);
            }
            max_depth++;
        }
        return max_depth;
    }

    /**
     * 剑指 Offer 55 - II. 平衡二叉树 - (简单)
     * https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/
     * @param root 二叉树根节点
     * @return 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。
     * 如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
     * 注意: 1 <= 树的结点个数 <= 10000
     */
    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        //自底向上递归树的深度: 时间复杂度O(N),空间复杂度O(N)
        return isBalancedDepth(root) != -1;
        //自顶向下递归树的深度: 时间复杂度O(NlogN(需要递归下去检查子树)),空间复杂度O(N)
        //return Math.abs(maxDepth(root.left) - maxDepth(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }
    private int isBalancedDepth(TreeNode root) {
        //本质就是进行后序遍历
        if (root == null) return 0;
        int left = isBalancedDepth(root.left);
        int right = isBalancedDepth(root.right);
        //（如果左右子树已存在不平衡情况,或者左右子树高度差超过1）均返回-1
        if (left == -1 || right == -1 || Math.abs(left-right) > 1) return -1;
        //如果左右子树高度差没超过1,返回较高者
        return Math.max(left,right) + 1;
    }

    /**
     * 剑指 Offer 28. 对称的二叉树 - (简单)
     * https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/
     * @param root 二叉树根节点
     * @return 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
     *         注意: 0 <= 节点个数 <= 1000
     */
    public boolean isSymmetric(TreeNode root) {
        return isSymmetric(root,root);
    }
    public boolean isSymmetric(TreeNode root1,TreeNode root2) {
        //到此说明两个都到底了都为null,对称
        if (root1 == null && root2 == null) return true;
        //到此说明两个之中有一个为null一个不为null,不对称
        if (root1 == null || root2 == null) return false;

        //到此说明两个的值不相同,不对称
        if (root1.val != root2.val) return false;

        //对左右子树进行对称判断,条件为: root1.left==root2.right && root1.right==root2.left
        return isSymmetric(root1.left, root2.right) && isSymmetric(root1.right, root2.left);
    }

    /**
     * 剑指 Offer 37. 序列化二叉树 - (困难)
     * https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/
     * 请实现两个函数，分别用来序列化和反序列化二叉树。
     */
    static class Codec {
        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            StringBuilder stb = new StringBuilder();
            stb.append("[");

            if (root != null) {
                Queue<TreeNode> queue = new LinkedList<>();
                queue.offer(root);
                //层次遍历添加节点
                while (!queue.isEmpty()){
                    TreeNode node = queue.poll();
                    if (node == null) stb.append("null");
                    else {
                        stb.append(node.val);
                        queue.offer(node.left);
                        queue.offer(node.right);
                    }
                    stb.append(",");
                }
                //在字符串中从后往前找最后一个节点
                for (int last = stb.length()-1; last > 0; last--) {
                    if (stb.charAt(last) == ',' && stb.charAt(last-1) != 'l') {
                        stb.delete(last,stb.length());
                        break;
                    }
                }
            }

            stb.append("]");
            return stb.toString();
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data == null || data.equals("[]")) return null;
            data = data.substring(1, data.length() - 1);

            String[] vals = data.split(",");
            TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
            int vals_idx = 1;

            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            //层次遍历添加节点
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();

                if (vals_idx < vals.length && !vals[vals_idx].equals("null")) {
                    node.left = new TreeNode(Integer.parseInt(vals[vals_idx]));
                    queue.offer(node.left);
                }
                vals_idx++;

                if (vals_idx < vals.length && !vals[vals_idx].equals("null")) {
                    node.right = new TreeNode(Integer.parseInt(vals[vals_idx]));
                    queue.offer(node.right);
                }
                vals_idx++;
            }

            return root;
        }

        // 递归版: Encodes a tree to a single string.
        public String serialize_recur(TreeNode root) {
            if (root == null) return "[]";

            String data = serialize_recur_helper(root);
            return "[" + data.substring(0, data.length() - 1) + "]";
        }
        public String serialize_recur_helper(TreeNode root) {
            if (root == null) return "null,";
            String str = root.val + ",";
            str += serialize_recur_helper(root.left);
            str += serialize_recur_helper(root.right);
            return str;
        }

        // 递归版: Decodes your encoded data to tree.
        public TreeNode deserialize_recur(String data) {
            if (data == null || data.equals("[]")) return null;
            data = data.substring(1, data.length() - 1);

            String[] vals = data.split(",");
            Queue<String> queue = new LinkedList<>();
            for (String val : vals) queue.offer(val);

            return deserialize_recur_helper(queue);
        }
        public TreeNode deserialize_recur_helper(Queue<String> queue) {
            String val = queue.poll();
            assert val != null;
            if (val.equals("null")) return null;

            TreeNode root = new TreeNode(Integer.parseInt(val));
            root.left = deserialize_recur_helper(queue);
            root.right = deserialize_recur_helper(queue);

            return root;
        }
    }

    /**
     * 剑指 Offer 54. 二叉搜索树的第k大节点 - (简单)
     * https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/
     * @param root 二叉搜索树根节点
     * @param k 1 ≤ k ≤ 二叉搜索树元素个数
     * @return 给定一棵二叉搜索树，请找出其中第k大的节点。
     */
    public int kthLargest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();

        //解决思路1: 如果利用二叉搜索树中序遍历,那么时间O(n),空间O(n维护一个集合)
        /*List<Integer> list = new ArrayList<>();
        while (!stack.isEmpty() || root != null) {
            //左路直下
            if (root != null) {
                stack.push(root);
                root = root.left;
            }
            else {
                root = stack.pop();
                list.add(root.val);
                root = root.right;
            }
        }
        return list.get(list.size()-k);*/

        //解决思路2: 利用二叉搜索树逆向中序遍历,那么时间O(n),空间O(1)
        int ith = 1;
        while (!stack.isEmpty() || root != null) {
            //右路直下
            if (root != null) {
                stack.push(root);
                root = root.right;
            }
            else {
                root = stack.pop();
                if (ith++ == k) return root.val;
                root = root.left;
            }
        }
        return -1;
    }

    /**
     * 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先 - (简单)
     * https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/
     * 最近公共祖先的定义为：对于有根树T的两个结点p和q，最近公共祖先表示为一个结点x，满足x是p和q的祖先且x的深度尽可能大。
     * （一个节点也可以是它自己的祖先）
     * @param root 二叉搜索树根节点
     * @param p 节点p
     * @param q 节点q
     * @return 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
     * 注意: 所有节点的值都是唯一的。
     *      p和q 为不同节点且均存在于给定的二叉搜索树中。
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (p == root || q == root) return root;

        //方法1: 通过不断上溯找共同的父节点,时间O(n^2) 空间O(n)
        /*while (true) {
            if (findTreeNode(p,q)) return p;
            p = getParent(root, p);

            if (findTreeNode(q,p)) return q;
            q = getParent(root, q);
        }*/

        //方法2: 利用二叉搜索树的性质缩小范围,左边比当前节点小,右边比当前节点大
        if (root.val > p.val && root.val > q.val)
            return lowestCommonAncestor(root.left,p,q);
        else if (root.val < p.val && root.val < q.val)
            return lowestCommonAncestor(root.right, p, q);
        //说明pq各在一边,这个就是最近公共祖先
        else return root;
    }
    private boolean findTreeNode(TreeNode parent, TreeNode node) {
        //在一个父节点中找指定子节点
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = parent;
        while (!stack.isEmpty() || cur != null) {
            if (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            else {
                cur = stack.pop();
                if (cur == node) return true;
                cur = cur.right;
            }
        }
        return false;
    }
    private TreeNode getParent(TreeNode root, TreeNode node) {
        //获取一个节点的父节点
        if (root == null || node == null) return null;
        if (root == node) return root;

        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (!stack.isEmpty() || cur != null) {
            if (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            else {
                cur = stack.pop();
                if (cur.left == node || cur.right == node) return cur;
                cur = cur.right;
            }
        }
        return null;
    }

    /**
     * 剑指 Offer 68 - II. 二叉树的最近公共祖先 - (简单)
     * @param root 二叉树根节点
     * @param p 节点p
     * @param q 节点q
     * @return 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
     * 注意: 所有节点的值都是唯一的。
     *      p和q 为不同节点且均存在于给定的二叉树中。
     */
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (p == root || q == root) return root;

        //方法1: 通过不断上溯找共同的父节点,时间O(n^2) 空间O(n)
        /*while (true) {
            if (findTreeNode(p,q)) return p;
            p = getParent(root, p);

            if (findTreeNode(q,p)) return q;
            q = getParent(root, q);
        }*/

        //方法2: 自顶向下递归寻找pq分布情况进行分析,时间O(n) 空间O(n)
        TreeNode left = lowestCommonAncestor2(root.left, p, q);
        TreeNode right = lowestCommonAncestor2(root.right, p, q);

        if (right == null) return left; //1.pq在左
        if (left == null) return right; //2.pq在右
        return root;                    //3.pq各一边
    }

}

/**二叉树数据结构**/
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int val) { this.val = val; }
}
