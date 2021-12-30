package 剑指_Offer;

import java.util.*;

/**
 * 具体算法类题目
 */
public class Algorithm_lcof {

    /**
     * 剑指 Offer 10- II. 青蛙跳台阶问题 - (简单)
     * https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/
     * @param n 0 <= n <= 100
     * @return 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。
     * 求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
     * 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
     */
    public int numWays(int n) {
        if (n == 0) return 1;

        int[] dp = new int[n+1];
        //base case
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            //状态转移方程: 1.从前一个台阶跳一步上来  2.从前两个台阶跳两步上来
            //可以进行状态压缩
            dp[i] = dp[i-1] + dp[i-2];
            dp[i] %= 1000000007;
        }

        return dp[n];
    }
    /*青蛙跳台阶问题升级版: 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法*/
    public int numWaysII(int n) {
        //这次0级台阶,青蛙不能跳
        if (n == 0) return 0;

        int[] dp = new int[n+1];
        //base case
        dp[0] = 0;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            //状态转移方程: 1.从前一个台阶跳一步上来  2.从前两个台阶跳两步上来   ....   n.从前n个台阶跳n步上来
            //因为: f(n-1) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2)
            //所以: f(n) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) + f(n-1) = f(n-1) + f(n-1) = 2 * f(n-1)
            dp[i] = dp[i-1] << 1;
            dp[i] %= 1000000007;
        }

        return dp[n];
    }

    /**
     * 剑指 Offer 13. 机器人的运动范围 - (中等)
     * https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/
     * 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。
     * 一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。
     * @param m 1 <= n,m <= 100
     * @param n 1 <= n,m <= 100
     * @param k 0 <= k <= 20
     * @return 请问该机器人能够到达多少个格子？
     */
    public int movingCount(int m, int n, int k) {
        //标记是否访问过
        boolean[][] coordinates = new boolean[m][n];

        /*深度优先:DFS*/
        //movingDFS(m,n,0,0,coordinates,k);

        /*广度优先:BFS*/
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{0,0});
        while (!queue.isEmpty()) {
            int[] xy = queue.poll();
            int x = xy[0];
            int y = xy[1];
            if (x < 0 || y < 0 || x >= m || y >= n || coordinates[x][y] || getDigitSum(x, y) > k) continue;
            count++;
            coordinates[x][y] = true;
            queue.offer(new int[]{x-1,y});  //左
            queue.offer(new int[]{x+1,y});  //右
            queue.offer(new int[]{x,y+1});  //上
            queue.offer(new int[]{x,y-1});  //下
        }

        return count;
    }
    private int count = 0;
    private void movingDFS(int m, int n, int x, int y, boolean[][] isVisited, int k) {
        if (x < 0 || y < 0 || x >= m || y >= n || isVisited[x][y] || getDigitSum(x, y) > k) return;
        count++;
        isVisited[x][y] = true;
        movingDFS(m, n, x-1, y, isVisited, k);    //左
        movingDFS(m, n, x+1, y, isVisited, k);    //右
        movingDFS(m, n, x, y+1, isVisited, k);    //上
        movingDFS(m, n, x, y-1, isVisited, k);    //下
    }
    private int getDigitSum(int x, int y) {
        int sum = 0;

        while (x != 0) {
            sum += x % 10;
            x = x / 10;
        }

        while (y != 0) {
            sum += y % 10;
            y = y / 10;
        }

        return sum;
    }

    // 剑指 Offer 12. 矩阵中的路径
    // 给定一个m x n 二维字符网格board 和一个字符串单词word 。如果word 存在于网格中，返回 true ；否则，返回 false 。
    //单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。
    // 同一个单元格内的字母不允许被重复使用。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0
                || word == null || word.length() == 0) {
            return false;
        }
        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == word.charAt(0) && existHelper(board, i, j, word, 0, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean existHelper(char[][] board, int row, int col, String word, int idx, boolean[][] visited) {
        if (row < 0 || col < 0 || row >= board.length || col >= board[0].length || visited[row][col]) {
            return false;
        }
        if (board[row][col] != word.charAt(idx)) {
            return false;
        }
        if (idx == word.length()-1) {
            return true;
        }
        visited[row][col] = true;
        boolean ans = existHelper(board, row+1, col, word, idx+1, visited)
                || existHelper(board, row-1, col, word, idx+1, visited)
                || existHelper(board, row, col+1, word, idx+1, visited)
                || existHelper(board, row, col-1, word, idx+1, visited);
        visited[row][col] = false;
        return ans;
    }

    /**
     * 剑指 Offer 46. 把数字翻译成字符串 - (中等)
     * https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/
     * 给定一个数字，我们按照如下规则把它翻译为字符串：
     *  0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
     *  一个数字可能有多个翻译。
     * @param num 0 <= num < 2^31
     * @return 请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
     */
    public int translateNum(int num) {
        //个位数只有一种解码方式
        if(num < 10) return 1;

        String nums = String.valueOf(num);

        //return traceback_transNum(nums,0);
        return dp_transNum(nums);
    }
    //解法一： 回溯算法
    public int traceback_transNum(String strs, int cur) {
        if (cur >= strs.length()) {
            return 1;
        }
        //如果能够解码2位,需要考虑 '0x' 这种特殊情况
        if (cur+2 <= strs.length() && strs.charAt(cur) != '0'
                && Integer.parseInt(strs.substring(cur,cur+2)) < 26) {
            return traceback_transNum(strs,cur+1) + traceback_transNum(strs,cur+2);
        }
        //如果只能解码1位
        return traceback_transNum(strs,cur+1);
    }
    //解法二： 动态规划
    public int dp_transNum(String strs) {
        //定义动态规划数组,代表字符串中子串能有多少种解法
        int[] dp = new int[strs.length() + 1];
        dp[0] = dp[1] = 1; //空子串和单位子串只有一种解法
        for (int i = 2; i <= strs.length(); i++) {
            String tmpStr = strs.substring(i - 2, i);
            //如果当前子串能够通过子子串解码1位-2位得到
            if (tmpStr.compareTo("10") >= 0 && tmpStr.compareTo("25") <= 0) {
                dp[i] = dp[i - 1] + dp[i - 2];
            }
            //如果当前子串只能通过子子串解码1位得到
            else {
                dp[i] = dp[i - 1];
            }
        }

        return dp[strs.length()];
    }

    //8. 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。
    // 数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。
    // 请找出数组中任意一个重复的数字。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof
    public int findRepeatNumber(int[] nums) {
        if(nums==null || nums.length==0) return -1;
        //解法一 哈希 时间O(N) 空间O(N)
        /*Set<Integer> sets = new HashSet<>();
        for (int num : nums) {
            if (sets.contains(num)) {
                return num;
            }
            sets.add(num);
        }*/
        //解法二 块排 时间O(NLogN) 空间O(1)
        /*Arrays.sort(nums);
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] == nums[i+1]) {
                return nums[i];
            }
        }*/
        //解法三 辅助数组 时间O(N) 空间O(N)
        /*boolean[] ints = new boolean[nums.length];
        for (int num : nums) {
            if (ints[num]) {
                return num;
            }
            ints[num] = true;
        }*/
        //解法四 原地哈希 时间O(N) 空间O(1)
        /*for(int i = 0 ; i < nums.length; i++){
            while(nums[i] != i){
                if(nums[i] == nums[nums[i]]){
                    return nums[i];
                }
                int temp = nums[nums[i]];
                nums[nums[i]] = nums[i];
                nums[i] = temp;
            }
        }
        return -1;*/
        //解法五 原地辅助数组 时间O(N) 空间O(1)
        for(int i : nums){
            int n = Math.abs(i);
            if(nums[n] < 0) return n;
            else nums[n] *= -1;
        }
        return 0;
    }

    //9.一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof
    public int missingNumber(int[] nums) {
        //解法一 遍历 时间O(N) 空间O(1)
        /*for (int i = 0; i < nums.length; i++) {
            if (i != nums[i]) {
                return i;
            }
        }
        return nums.length;*/

        //解法二 二分查找 时间O(NLogN) 空间O(1)
        int l = 0, r = nums.length-1;
        while (l < r) {
            int m = (l + r) >> 1;
            if (nums[m] == m) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        return l == nums[l] ? nums.length : l;
    }

    //10.在排序数组中统计一个数字在排序数组中出现的次数。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/
    public int search(int[] nums, int target) {
        //解法一 遍历 时间O(N) 空间O(1)
        int count = 0;
        /*for (int i = 0; i < nums.length; i++) {
            if (nums[i] > target) {
                break;
            }
            if (nums[i] == target) {
                while (i < nums.length && nums[i++] == target) {
                    count++;
                }
            }
        }*/

        //解法二 二分查找 时间O(NLogN) 空间O(1)
        int l = 0, r = nums.length-1;
        while (l < r) {
            int m = (l + r) >> 1;
            if (nums[m] >= target) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        while (l < nums.length && nums[l++] == target) {
            count++;
        }
        return count;
    }

    // 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。
    // 求所有子数组的和的最大值。
    // 要求时间复杂度为O(n)。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        //解法一 动态规划 时间O(N) 空间O(N)
        /*int[] dp = new int[nums.length];
        dp[0] = nums[0];
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            //判断是否要与下一个数值组队
            dp[i] = Math.max(dp[i-1] + nums[i], nums[i]);
            //记录最大的子数组的和
            max = Math.max(max, dp[i]);
        }*/

        //解法二 动态规划+状态压缩 时间O(N) 空间O(1)
        int sum = nums[0];
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            //判断是否要与下一个数值组队
            sum = Math.max(sum + nums[i], nums[i]);
            //记录最大的子数组的和
            max = Math.max(max, sum);
        }
        return max;
    }

    // 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。
    // 你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。
    // 给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof
    public int maxValue(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        int m = grid.length;
        int n = grid[0].length;

        //解法一 DFS 最大利益驱使 时间复杂度O(nm^2)
        //return maxValueHelper(grid, 0, 0);

        //解法二 动态规划 转移方程: f(i, j) = max{f(i - 1, j), f(i, j - 1)} + grid_current_value
        //dp[i][j] 表示从左上角(0,0)以任意移动方式到达当前点(i,j)的最大价值
        //时间 O(nm) 空间O(nm)
        /*int[][] dp = new int[m + 1][n + 1];
        for(int i = 1; i <= m; i++) {
            for(int j = 1; j <= n; j++) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1];
            }
        }
        return dp[m][n];*/

        //解法三 动态规划 - 状态压缩 时间 O(nm) 空间O(n)
        int[] dp = new int[n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[j] = Math.max(dp[j], dp[j - 1]) + grid[i - 1][j - 1];
            }
        }

        return dp[n];
    }

    private int maxValueHelper(int[][] grid, int mi, int ni) {
        if (mi < 0 || ni < 0) {
            return 0;
        }
        if (mi >= grid.length || ni >= grid[0].length) {
            return 0;
        }

        return grid[mi][ni] + Math.max(maxValueHelper(grid, mi + 1, ni), maxValueHelper(grid, mi, ni + 1));
    }

    // 剑指 Offer 48. 最长不含重复字符的子字符串
    // 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        if (s.length() == 1) {
            return 1;
        }

        int max = Integer.MIN_VALUE;

        //解法一 使用双端队列维护一个不存在重复值的滑动窗口 时间O(N^2) 空间 O(N)
        /*Deque<Character> deque = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            //如果当前队列中不存在重复字符,正常入队
            if (!deque.contains(c)) {
                deque.add(c);
            }
            //否则需要左端剔除重复值往前的所有字符
            else {
                max = Math.max(max, deque.size());
                while(deque.getFirst() != s.charAt(i)) {
                    deque.removeFirst();
                }
                deque.removeFirst();
                deque.addLast(s.charAt(i));
            }
        }
        return Math.max(max, deque.size());*/
        //解法二 优化使用Set维护一个不存在重复值的滑动窗口 时间O(N) 空间 O(1)
        /*Set<Character> set = new HashSet<>();  //Set空间最多就O(128),同理如下
        int lastIdx = 0;    //记录历史上一次出现重复字符的位置
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            while (set.contains(c)) {
                set.remove(s.charAt(lastIdx++));
            }
            set.add(c);
            max = Math.max(max, i - lastIdx + 1);
        }*/

        //解法三 动态规划 + 线性查找 时间O(N^2) 空间 O(1)
        /*int[] dp = new int[s.length()];
        dp[0] = 1;
        for (int i = 1; i < s.length(); i++) {
            int j = i - 1;
            while(j >= 0 && s.charAt(i) != s.charAt(j)) j--; // 线性查找
            dp[i] = dp[i - 1] < i - j ? dp[i - 1] + 1 : i - j; // 无法理解???
            max = Math.max(max, dp[i]); // max(dp[i - 1], dp[i])
        }*/

        //解法四 动态规划 + 哈希映射<字符,该字符在字符串中上一次出现位置> 时间O(N) 空间 O(1)
        /*Map<Character, Integer> map = new HashMap<>();
        int[] dp = new int[s.length()];
        dp[0] = 1;
        map.put(s.charAt(0), 0);
        for (int i = 1; i < s.length(); i++) {
            char c = s.charAt(i);
            int j = map.getOrDefault(c, -1);
            dp[i] = dp[i - 1] < i - j ? dp[i - 1] + 1 : i - j;  // 无法理解???
            //更新该字符在字符串中上一次出现位置
            map.put(c, i);
            max = Math.max(max, dp[i]); // max(dp[i - 1], dp[i])
        }*/

        //解法五 动态规划状态压缩成双指针 + 哈希映射<字符,该字符在字符串中上一次出现位置> 时间O(N) 空间 O(1)
        Map<Character, Integer> map = new HashMap<>();
        int j = -1; //记录历史上一次出现重复字符的位置
        for(int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if(map.containsKey(c)) {
                j = Math.max(j, map.get(c)); // 更新左指针
            }
            //更新该字符在字符串中上一次出现位置
            map.put(c, i);
            max = Math.max(max, i - j); // max(dp[i - 1], dp[i])
        }

        return max;
    }

    // 剑指 Offer 45. 把数组排成最小的数
    // 输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，
    // 打印能拼接出的所有数字中最小的一个。
    public String minNumber(int[] nums) {
        //数组中两两组合排序判断规则： x+y>y+x 则 "xy" 大于 "yx"
        List<String> numStrs = new ArrayList<>(nums.length);
        for (int num : nums) {
            numStrs.add(String.valueOf(num));
        }
        numStrs.sort((o1, o2) -> (o1 + o2).compareTo(o2 + o1));
        StringBuilder stb = new StringBuilder();
        for (String numStr : numStrs) {
            stb.append(numStr);
        }
        return stb.toString();
    }

    // 剑指 Offer 61. 扑克牌中的顺子
    // 从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。
    // 2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。
    // A 不能视为 14。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof
    public boolean isStraight(int[] nums) {
        if (nums == null || nums.length != 5) {
            return false;
        }
        Arrays.sort(nums);
        //顺子达成条件:
        //1. 数与数之间的差值不超过5
        //2. 不能有不为0的相同数
        int diff = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i-1] == 0) {
                continue;
            }
            if (nums[i] == nums[i-1]) {
                return false;
            }
            diff += nums[i] - nums[i-1];
        }
        return diff < 5;
    }

    // 剑指 Offer 40. 最小的k个数
    // 输入整数数组 arr ，找出其中最小的 k 个数。
    // 例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
    public int[] getLeastNumbers(int[] arr, int k) {
        if (arr == null || arr.length == 0 || k == 0) {
            return new int[0];
        }
        if (k == arr.length) {
            return arr;
        }
        int[] result = new int[k];

        //解法一 排序后取前k个 时间O(NLogN) 空间O(K)
        /*Arrays.sort(arr);
        System.arraycopy(arr, 0, result, 0, k);*/

        //解法二 大根堆排 时间O(NLogN) 空间O(K)
        /*PriorityQueue<Integer> maxHeap = new PriorityQueue<>((o1, o2) -> o2 - o1);
        for (int i : arr) {
            if (maxHeap.size() < k) {
                maxHeap.offer(i);
            } else if (maxHeap.peek() > i) {
                maxHeap.poll();
                maxHeap.offer(i);
            }
        }
        int idx = 0;
        for (Integer i : maxHeap) {
            arr[idx++] = i;
        }*/

        //解法二 切分快排 时间O(NLogN) 空间O(K)
        morphedQuickSort(arr, 0, arr.length-1, result, k-1);
        return result;
    }

    private void morphedQuickSort(int[] arr, int begin, int end, int[] result, int k) {
        int mid = findQuickSortPivot(arr, begin, end);
        if (mid == k) {
            System.arraycopy(arr, 0, result, 0, k+1);
            return;
        }
        if (mid > k) {
            morphedQuickSort(arr, begin, mid - 1, result, k);
        } else {
            morphedQuickSort(arr, mid + 1, end, result, k);
        }
    }

    private int findQuickSortPivot(int[] arr, int begin, int end) {
        int pivot = arr[begin];
        int i = begin, j = end + 1;
        while (true) {
            while (++i <= end && arr[i] < pivot);
            while (--j >= begin && arr[j] > pivot);
            if (i >= j) break;
            int tmp = arr[j];
            arr[j] = arr[i];
            arr[i] = tmp;
        }
        arr[begin] = arr[j];
        arr[j] = pivot;
        return j;
    }

    // 剑指 Offer 41. 数据流中的中位数
    // 如何得到一个数据流中的中位数？
    // 如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
    // 如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
    //例如 [2,3,4]的中位数是 3
    //    [2,3] 的中位数是 (2 + 3) / 2 = 2.5
    //设计一个支持以下两种操作的数据结构：
    //void addNum(int num) - 从数据流中添加一个整数到数据结构中。
    //double findMedian() - 返回目前所有元素的中位数。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof
    static class MedianFinder {
        Queue<Integer> minHeap, maxHeap;
        /** initialize your data structure here. */
        public MedianFinder() {
            minHeap = new PriorityQueue<>(); // 小顶堆，保存较大的一半
            maxHeap = new PriorityQueue<>((x, y) -> (y - x)); // 大顶堆，保存较小的一半
        }
        public void addNum(int num) {
            if(minHeap.size() != maxHeap.size()) {
                minHeap.add(num);
                maxHeap.add(minHeap.poll());
            } else {
                maxHeap.add(num);
                minHeap.add(maxHeap.poll());
            }
        }
        public double findMedian() {
            return minHeap.size() != maxHeap.size() ? minHeap.peek() : (minHeap.peek() + maxHeap.peek()) / 2.0;
        }
    }

    // 剑指 Offer 64. 求1+2+…+n
    // 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/qiu-12n-lcof/
    public int sumNums(int n) {
        //等差数列求和公式 Sn=n(a1+an)/2 由于a1为1
        //即可化简为 Sn=n(1+n)>>1=(n+n^2)>>1
        return (int) (Math.pow(n,2) + n)  >> 1;
    }

    // 剑指 Offer 16. 数值的整数次方
    // 实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，x^n）。
    // 不得使用库函数，同时不需要考虑大数问题。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/
    public double myPow(double x, int n) {
        //解法一 快速幂 递归版
        /*if(n == 0) return 1;
        if(n == 1) return x;
        if(n == -1) return 1 / x;
        double half = myPow(x, n / 2);
        double mod = myPow(x, n % 2);
        return half * half * mod;*/

        //解法二 快速幂 迭代版
        double res = 1;
        double base = x;
        boolean flag = n >= 0;
        //负数取反，考虑到最小负数，需要先自增，后续再除以2
        if(!flag) n = -(++n);
        while(n > 0) {
            if((n & 1) == 1) res *= x;
            n >>= 1;
            x *= x;
        }
        return flag ? res :1 / (res * base);
    }

    // 剑指 Offer 15. 二进制中1的个数
    // 编写一个函数，输入是一个无符号整数（以二进制串的形式），
    // 返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量).）。
    // 提示1：
    // 请注意，在某些语言（如 Java）中，没有无符号整数类型。
    // 在这种情况下，输入和输出都将被指定为有符号整数类型，
    // 并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
    // 在 Java 中，编译器使用 二进制补码 记法来表示有符号整数。
    // 因此，在上面的示例 3中，输入表示有符号整数 -3。
    // 提示2：
    // 输入必须是长度为 32 的 二进制串 。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof
    public int hammingWeight(int n) {
        //解法1 将数字转成二进制串后计数
        /*boolean flag = n >= 0;
        n = Math.abs(n);
        StringBuilder stb = new StringBuilder();
        while (n > 0) {
            stb.insert(0,n%2);
            n >>= 1;
        }
        //长度必须为32位的二进制串
        while (stb.length() < 32) {
            stb.insert(0,'0');
        }
        //负数需要以补码方式表示
        if (!flag) {
            for (int i = 0; i < stb.length(); i++) {
                if (stb.charAt(i) == '1') stb.replace(i,i+1,"0");
                else stb.replace(i,i+1,"1");
            }
            int i = stb.length()-1;
            for (; i >= 0; i--) {
                if (stb.charAt(i) == '0') {
                    stb.replace(i,i+1,"1");
                    break;
                }
            }
            if (i < 0) {
                stb.insert(0,'1');
                i++;
            }
            while (++i < stb.length()) {
                stb.replace(i,i+1,"0");
            }
        }
        int count = 0;
        for (int i = 0; i < stb.length(); i++) {
            if (stb.charAt(i) == '1') {
                count++;
            }
        }
        return count;*/

        //解法二 使用Integer提供的API
        //return Integer.bitCount(n);

        //解法三 位移取最右1
        /*int count = 0;
        while (n != 0) {
            if ((n & 1) == 1) count++;
            n >>>= 1;
        }
        return count;*/

        //解法四 位运算
        int count = 0;
        while (n != 0) {
            n &= (n-1);
            count++;
        }
        return count;
    }

    // 剑指 Offer 65. 不用加减乘除做加法
    // 写一个函数，求两个整数之和，
    // 要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。
    // 提示： a, b 均可能是负数或 0 但是结果不会溢出 32 位整数
    public int add(int a, int b) {
        if (a == 0) return b;
        if (b == 0) return a;
        if (a == b) return a << 1;

        //解法一 将数字转成二进制串后进行进制计算
        /*StringBuilder aBits = new StringBuilder(Integer.toBinaryString(a));
        while (aBits.length() < 32) {
            aBits.insert(0, '0');
        }
        StringBuilder bBits = new StringBuilder(Integer.toBinaryString(b));
        while (bBits.length() < 32) {
            bBits.insert(0, '0');
        }
        StringBuilder result = new StringBuilder(32);
        char x = '0';
        for (int i = 31; i >= 0; i--) {
            char ac = aBits.charAt(i);
            char bc = bBits.charAt(i);
            if (ac == bc) {
                result.insert(0, x);
                x = ac;
                continue;
            }
            result.insert(0, x == '1' ? '0':'1');
        }
        return Integer.parseUnsignedInt(result.toString(),2);*/

        //解法二 使用异或模拟加法运算
        while(b != 0) {
            //计算进位数
            int x = (a & b) << 1;
            //异或记录无需进位的数
            a ^= b;
            //用b保存需要加在答案上的进位数
            b = x;
        }
        return a;
    }

    // 剑指 Offer 62. 圆圈中最后剩下的数字
    // 0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。
    // 求出这个圆圈里剩下的最后一个数字。
    // 限制： 1 <= n <= 10^5 && 1 <= m <= 10^6
    // 来源：力扣（LeetCode）
    // 链接：https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof
    public int lastRemaining(int n, int m) {
        //解法一 模拟删除 时间O(n) 空间O(n)
        /*ArrayList<Integer> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(i);
        }
        int idx = 0;
        while (n > 1) {
            idx = (idx+m-1)%n;
            list.remove(idx);
            n--;
        }
        return list.get(0);*/

        //解法二 公式法 时间O(n) 空间O(1)
        //f(N,M)=(f(N−1,M)+M)%N
        //当只有一人时,答案下标为0
        int ans = 0;
        for (int i = 2; i <= n; i++) {
            ans = (ans + m) % i;
        }
        return ans;
    }

    // 剑指 Offer 57 - II. 和为s的连续正数序列
    // 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
    // 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
    // 限制： 1 <= target <= 10^5
    // 来源：力扣（LeetCode）
    // 链接：https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof
    public int[][] findContinuousSequence(int target) {
        if (target <= 2) return new int[0][0];
        //解法一 遍历
        /*List<int[]> tmp = new ArrayList<>();
        for (int i = 1; i < target-1; i++) {
            int sum = i;
            for (int j = i+1; j < target; j++) {
                sum += j;
                if (sum > target) {
                    break;
                }
                if (sum == target && j-i+1 >= 2) {
                    int[] seq = new int[j-i+1];
                    for (int k = i; k <= j; k++) {
                        seq[k-i] = k;
                    }
                    tmp.add(seq);
                    break;
                }
            }
        }*/

        //解法二 双指针滑动窗口
        List<int[]> tmp = new ArrayList<>();
        for (int l = 1,r = 1,sum = 0; r < target; r++) {
            sum += r;
            while (sum > target) {
                sum -= l++;
            }
            if (sum == target && r-l+1 >= 2) {
                int[] seq = new int[r-l+1];
                for (int i = 0; i < seq.length; i++) {
                    seq[i] = i+l;
                }
                tmp.add(seq);
            }
        }

        int[][] result = new int[tmp.size()][];
        for (int i = 0; i < result.length; i++) {
            result[i] = tmp.get(i);
        }
        return result;
    }

    // 剑指 Offer 14- I. 剪绳子
    // 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。
    // 请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
    // 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
    // 提示： 2 <= n <= 58
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/jian-sheng-zi-lcof
    public int cuttingRope(int n) {
        if(n <= 3) return n - 1;
        /*
            解法一：动态规划
            当 2 ≤ n ≤ 58 时，假设对正整数 n 拆分出的第一个正整数是 j（1 < j < n），则有以下两种方案：
            将 n 拆分成 j 和 n-j 的和，且 n-j 不再拆分成多个正整数，此时的乘积是 j*(n-j);
            将 n 拆分成 j 和  n-j 的和，且  n-j 继续拆分成多个正整数，此时的乘积是 j*dp[n−j];
            判断这两种方案谁较优即可。
        */
        /*int[] dp = new int[n+1];
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                dp[i] = Math.max(Math.max(j*(i-j), j*dp[i-j]), dp[i]);
            }
        }
        return dp[n];*/

        /*
            解法二 贪心算法 尽可能将绳子以长度3等分为多段时，乘积最大
            n = 3a+b   b属于{0,1,2}
            最优： 3a+0=n 把绳子尽可能切为多个长度为 3 的片段
            次优： 3a+2=n 若最后一段绳子长度为2则保留,不再拆为
            最差： 3(a-1)+2+2=n 若最后一段绳子长度为1则拆掉一个3拼成 3+1 -> 2+2
         */
        int a = n / 3, b = n % 3;
        if(b == 0) return (int)Math.pow(3, a);
        if(b == 1) return (int)Math.pow(3, a - 1) * (2 + 2);
        return (int)Math.pow(3, a) * 2;
    }

    // 剑指 Offer 38. 字符串的排列
    // 输入一个字符串，打印出该字符串中字符的所有排列。
    // 限制： 1 <= s 的长度 <= 8
    // 来源：力扣（LeetCode）
    // 链接：https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/
    public String[] permutation(String s) {
        if (s == null || s.length() == 0) return new String[0];
        if (s.length() == 1) return new String[]{s};
        boolean[] visited = new boolean[s.length()];
        StringBuilder stb = new StringBuilder();
        //解法一 全排列后通过Set去重
        //Set<String> result = new HashSet<>();
        //permutation_trace_back_helper(s, visited, result, stb);
        //解法二 全排列+排序剪枝
        List<String> result = new ArrayList<>();
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        permutation_trace_back_helper2(chars, visited, result, stb);
        return result.toArray(new String[0]);
    }

    private void permutation_trace_back_helper(String s, boolean[] visited, Set<String> result, StringBuilder stb) {
        if (stb.length() == s.length()) {
            result.add(stb.toString());
            return;
        }
        for (int i = 0; i < s.length(); i++) {
            if (visited[i]) continue;
            visited[i] = true;
            stb.append(s.charAt(i));
            permutation_trace_back_helper(s, visited, result, stb);
            stb.deleteCharAt(stb.length()-1);
            visited[i] = false;
        }
    }

    private void permutation_trace_back_helper2(char[] chars, boolean[] visited, List<String> result, StringBuilder stb) {
        if (stb.length() == chars.length) {
            result.add(stb.toString());
            return;
        }
        for (int i = 0; i < chars.length; i++) {
            // visited[i-1] == false,说明同一树层chars[i-1]被排列过
            if (i > 0 && !visited[i-1] && chars[i] == chars[i-1]) continue;
            if (visited[i]) continue;
            visited[i] = true;
            stb.append(chars[i]);
            permutation_trace_back_helper2(chars, visited, result, stb);
            stb.deleteCharAt(stb.length()-1);
            visited[i] = false;
        }
    }

    // 剑指 Offer 49. 丑数
    // 我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。
    // 求按从小到大的顺序的第 n 个丑数。
    // 说明: 1 是丑数。
    // n 不超过1690。
    // 来源：力扣（LeetCode）
    // 链接：https://leetcode-cn.com/problems/chou-shu-lcof/
    public int nthUglyNumber(int n) {
        //解法一 遍历判断
        /*int idx = 1;
        int idn = 1;
        while (idx < n) {
            int num = ++idn;
            while (num % 2 == 0) num /= 2;
            while (num % 3 == 0) num /= 3;
            while (num % 5 == 0) num /= 5;
            if (num == 1) idx++;
        }
        return idn;*/

        //解法二 大丑数=小丑数*(2|3|5),1th丑数=1
        int[] dp = new int[n+1];
        dp[0] = 1;
        for (int a = 0,b = 0,c = 0, i = 1; i < n; i++) {
            int uglyA = dp[a]*2;
            int uglyB = dp[b]*3;
            int uglyC = dp[c]*5;
            dp[i] = Math.min(Math.min(uglyA, uglyB), uglyC);
            if (uglyA == dp[i]) a++;
            if (uglyB == dp[i]) b++;
            if (uglyC == dp[i]) c++;
        }
        return dp[n-1];
    }

}