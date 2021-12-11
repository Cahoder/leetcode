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

}