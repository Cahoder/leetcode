import java.util.*;

/**
 * leetcode 每日一题
 */
public class DailyQuestion {

    /**
     * 2020-11-5 单词接龙（利用图的广度优先遍历获取最短路经）
     * https://leetcode-cn.com/problems/word-ladder/comments/
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        //将字典放入哈希表中,并且判断是否包含endWord
        Set<String> wordSet = new HashSet<>(wordList);
        if (wordSet.size() < 1 || !wordSet.contains(endWord)) return 0;
        //防止wordSet中的包含着beginWord,需要去重
        wordSet.remove(beginWord);

        //使用一个队列和哈希表表示visited过的节点
        Queue<String> queue = new LinkedList<>();
        queue.offer(beginWord);
        Set<String> visited = new HashSet<>();
        visited.add(beginWord);

        //进行BFS广度优先搜寻图的最短路径
        int step = 1;
        while (!queue.isEmpty()){
            int currentSize = queue.size();
            for (int i = 0; i < currentSize; i++) {
                // 依次遍历当前队列中的单词
                String currentWord = queue.poll();
                // 如果 currentWord 能够修改 1 个字符与 endWord 相同，则返回 step + 1
                if (checkWordDistortOneLetter(currentWord, endWord, queue, visited, wordSet)) return step + 1;
            }
            step++;
        }
        return 0;
    }
    /**
     * 根据给定的currentWord获取其所有在wordSet中的变形(每次只能修改一个字符)
     * @param currentWord
     * @param endWord
     * @param queue
     * @param visited
     * @param wordSet
     * @return
     */
    private boolean checkWordDistortOneLetter(String currentWord, String endWord, Queue<String> queue,
                                               Set<String> visited, Set<String> wordSet){
        char[] chars = currentWord.toCharArray();
        for (int i = 0; i < endWord.length(); i++) {
            //获取currentWord对应endWord的每一个字符
            char c = chars[i];
            for (char j = 'a'; j < 'z'; j++) {
                if (c == j) continue;
                //修改单个字符,进行相关判断
                chars[i] = j;
                String nextWord = String.valueOf(chars);
                if (wordSet.contains(nextWord)){
                    //如果nextWord就是要找的endWord直接返回true
                    if (nextWord.equals(endWord)) return true;
                    //加入队列进行广度遍历,并且标记为visited
                    if (!visited.contains(nextWord)){
                        queue.offer(nextWord);
                        visited.add(nextWord);
                    }
                }
            }
            //切记不要忘了修改回来
            chars[i] = c;
        }
        return false;
    }

    /**
     * 2020-11-6 根据数字二进制下 1 的数目排序
     * https://leetcode-cn.com/problems/sort-integers-by-the-number-of-1-bits/
     * @param arr  1<= arr.length <= 500 && 0 <= arr[i] <= 10^4
     * @return 排序后的数组
     */
    public int[] sortByBits(int[] arr) {
        TreeMap<Integer,ArrayList<Integer>> map = new TreeMap<>();
        for (int value : arr) {
            int nums_1 = 0;
            //由于数据集最多也就到10^4,可以按位与运算获取相应的二进制位
            for (int bits = Integer.toBinaryString(10000).length(); bits >= 0; bits--) {
                if ((value >>> bits & 1) == 1) nums_1++;
            }
            //按照二进制下1的位数生成映射关系
            if (map.get(nums_1) == null) {
                ArrayList<Integer> item = new ArrayList<>();
                item.add(value);
                map.put(nums_1, item);
            } else map.get(nums_1).add(value);
        }
        int idx = 0;
        for (Integer nums_1 : map.keySet()) {
            ArrayList<Integer> alist = map.get(nums_1);
            //如果二进制下1的位数相同,需要再升序排序
            if(alist.size() > 1) Collections.sort(alist);
            for (Integer integer : alist) arr[idx++] = integer;
        }
        return arr;
    }

    /**
     * 2020-11-7
     * 给定一个整数数组nums，返回区间和在[lower, upper]之间的个数，包含lower和upper。
     * 区间和S(i, j)表示在nums中，位置从i到j的元素之和，包含i和j(i ≤ j)。
     * https://leetcode-cn.com/problems/count-of-range-sum/
     * @param nums
     * @param lower
     * @param upper
     * @return
     */
    public int countRangeSum(int[] nums, int lower, int upper) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            long sum = 0;
            for (int j = i; j < nums.length; j++) {
                sum += nums[j];
                if (lower <= sum && sum <= upper) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * 2020-11-8 买卖股票的最佳时机 II
     * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     * 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     * https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
     * @param prices 1 <= prices.length <= 3 * 10 ^ 4 并且 0 <= prices[i] <= 10 ^ 4
     * @return 最大收益
     */
    public int maxProfit2(int[] prices) {
        /*
        //贪心算法 --- 简化问题:只要今天的价格小于明天的价格,就买入然后明天卖出
        int profit = 0;
        for (int i = 0; i < prices.length-1; i++) {
            if (prices[i] < prices[i+1]) profit += prices[i+1] - prices[i];
        }
        return profit;
        */

        //动态规划
        //dp[i][0]表示第i天不持有股票的收益
        //dp[i][1]表示第i天持有股票的收益
        int[][] dp = new int[prices.length][2];
        dp[0][0] = 0;   //如果第一天不买不赚不赔
        dp[0][1] = -1 * prices[0];  //如果第一天买入肯定先负债
        for (int i = 1; i < prices.length; i++) {
            //第i天不持有股票:可能是前一天也不持有, 或者前一天卖出了. 看看哪个收益大?
            dp[i][0] = Math.max(dp[i-1][0],dp[i-1][1] + prices[i]);
            //第i天持有股票:可能是前一天也持有, 或者前一天没有卖出了. 看看哪个收益大?
            dp[i][1] = Math.max(dp[i-1][1],dp[i-1][0] - prices[i]);
        }
        //获取最后一天不持有股票的收益就是答案
        return dp[prices.length-1][0];
    }

    /**
     * 2020-11-9 最接近原点的 K 个点
     * 我们有一个由平面上的点组成的列表 points。需要从中找出 K 个距离原点 (0, 0) 最近的点
     * https://leetcode-cn.com/problems/k-closest-points-to-origin/
     * @param points 1 <= points.length <= 10000
     * @param K 1 <= K <= points.length
     *          && -10000 < points[i][0] < 10000
     *          && -10000 < points[i][1] < 10000
     * @return 最接近原点的 K 个点
     */
    public int[][] kClosest(int[][] points, int K) {
        //直接利用数组工具类排序接口
        Arrays.sort(points, new Comparator<int[]>() {
            public int compare(int[] point1, int[] point2) {
                return (point1[0] * point1[0] + point1[1] * point1[1]) - (point2[0] * point2[0] + point2[1] * point2[1]);
            }
        });
        return Arrays.copyOfRange(points, 0, K);

        /*
        if (points.length == K) return points;
        TreeMap<Integer,ArrayList<Integer>> map = new TreeMap<>();
        for (int i = 0; i < points.length; i++) {
            int Euclidean_distance = (int) (Math.pow(points[i][0],2) + Math.pow(points[i][1],2));
            map.computeIfAbsent(Euclidean_distance, k -> new ArrayList<>());
            //记录索引
            map.get(Euclidean_distance).add(i);
        }

        int[][] rst = new int[K][2];
        int idx = 0;
        for (Integer ed : map.keySet()) {
            for (Integer pidx : map.get(ed)) {
                if (idx == K) return rst;
                rst[idx++] = points[pidx];
            }
        }
        return rst;
        */
    }

    /**
     * 2020-11-10 实现获取下一个排列的函数,算法需要将给定数字序列重新排列成字典序中下一个更大的排列
     * https://leetcode-cn.com/problems/next-permutation/
     * @param nums 随机数组
     */
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length < 2) return;

        boolean flag = false;
        for (int i = nums.length-1; i > 0; i--) {
            if (nums[i] > nums[i-1]) {
                int tmp = nums[i-1];
                //从比他大的位数往后升序排
                Arrays.sort(nums,i,nums.length);
                for (int j = i; j < nums.length; j++) {
                    if (nums[j] > tmp) {
                        nums[i-1] = nums[j];
                        nums[j] = tmp;
                        break;
                    }
                }
                flag = true;
                break;
            }
        }
        //说明一开始就是降序排的,转成升序即可
        if (!flag) Arrays.sort(nums);
    }

    /**
     * 2020-11-11
     * 给定一个字符串 ring，表示刻在外环上的编码
     * 给定另一个字符串 key，表示需要拼写的关键词
     * 您需要算出能够拼写关键词中所有字符的最少步数
     * 两个字符串中都只有小写字符，并且均可能存在重复字符
     * https://leetcode-cn.com/problems/freedom-trail/
     * @param ring 外环上的编码 1 <= ring.length() <= 100
     * @param key 拼写的关键词 1 <= key.length() <= 100
     * @return 最少步数 字符串key一定可以由字符串ring旋转拼出
     */
    public int findRotateSteps(String ring, String key) {
        List<Integer>[] pos = new List[26];
        //开辟出26个字母字符,记录ring中每个字母在ring中出现的次数,保存其索引
        for (int i = 0; i < pos.length; i++) {
            pos[i] = new ArrayList<Integer>();
        }
        for (int i = 0; i < ring.length(); ++i) {
            pos[ring.charAt(i) - 'a'].add(i);
        }

        //表示在当前ring第k个字符与12:00方向对齐时第j个字符旋转到12:00方向并按下拼写的最少步数
        int[][] dp = new int[key.length()][ring.length()];
        //默认填充一个无穷大的数,https://blog.csdn.net/jiange_zh/article/details/50198097
        for (int i = 0; i < key.length(); ++i) Arrays.fill(dp[i], 0x3f3f3f);

        for (int i : pos[key.charAt(0) - 'a']) dp[0][i] = Math.min(i, ring.length() - i) + 1;

        for (int i = 1; i < key.length(); ++i) {
            for (int j : pos[key.charAt(i) - 'a']) {
                for (int k : pos[key.charAt(i - 1) - 'a']) {
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][k] + Math.min(Math.abs(j - k),
                            ring.length() - Math.abs(j - k)) + 1);
                }
            }
        }
        return Arrays.stream(dp[key.length()-1]).min().getAsInt();
    }

    /**
     * 2020-11-12 按奇偶排序数组 II
     * 给定一个非负整数数组A，A中一半整数是奇数，一半整数是偶数
     * 对数组进行排序，以便当A[i]为奇数时，i也是奇数；当A[i]为偶数时，i也是偶数
     * 你可以返回任何满足上述条件的数组作为答案
     * https://leetcode-cn.com/problems/sort-array-by-parity-ii/
     * @param A 待排序数组
     * 2 <= A.length <= 20000 && A.length % 2 == 0 && 0 <= A[i] <= 1000
     * @return 排序后数组
     */
    public int[] sortArrayByParityII(int[] A) {
        /*for (int i = 0; i < A.length;) {
            if ((i&1) == (A[i]&1)) {
                i++;
            } else {
                for (int j = i+1; j < A.length; j++) {
                    if ((i&1) == (A[j]&1)) {
                        int tmp = A[i];
                        A[i] = A[j];
                        A[j] = tmp;
                        break;
                    }
                }
            }
        }*/

        //双指针
        int slow = 1;
        for (int i = 0; i < A.length; i+=2) {
            if ((A[i]&1) == 1) {
                while ((A[slow]&1) == 1) slow+=2;
                //慢指针找到偶数下标
                int tmp = A[i];
                A[i] = A[slow];
                A[slow] = tmp;
            }
        }
        return A;
    }

    /**
     * 2020-11-13 奇偶链表
     * 把所有的奇数节点编号和偶数节点编号分别排在一起
     * 链表的第一个节点视为奇数节点 第二个节点视为偶数节点 以此类推
     * https://leetcode-cn.com/problems/odd-even-linked-list/
     * @param head 链表头
     * @return 修改后的链表头
     */
    public ListNode oddEvenList(ListNode head) {
        //如果节点数小于3直接返回即可
        if (head == null || head.next == null || head.next.next == null) return head;

        //奇数节点浮标
        ListNode oddCur = head;

        //偶数节点头部
        ListNode even = head.next;
        //偶数节点浮标
        ListNode evenCur = even;

        while (evenCur!=null && evenCur.next!=null) {
            oddCur.next = evenCur.next;
            oddCur = oddCur.next;

            evenCur.next = oddCur.next;
            evenCur = evenCur.next;
        }
        //将奇数和偶数链表头拼接
        oddCur.next = even;
        return head;
    }

    /**
     * 2020-11-14 数组的相对排序
     * 使 arr1 中项的相对顺序和 arr2 中的相对顺序相同
     * 未在 arr2 中出现过的元素需要按照升序放在 arr1 的末尾
     * https://leetcode-cn.com/problems/relative-sort-array/
     * @param arr1 对 arr1 中的元素进行排序
     * @param arr2 arr2 中的每个元素都出现在 arr1 中
     * @return 对arr1中的元素相对arr2排序后
     */
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        if (arr1 == null || arr1.length < 2) return arr1;

        /*
        //时间复杂度 O(n^2)
        //arr1目前需要交换位置
        int swap = 0;
        for (int value : arr2) {
            for (int j = swap; j < arr1.length; j++) {
                //找到需要交换的位置
                if (arr1[j] == value) {
                    int tmp = arr1[swap];
                    arr1[swap] = arr1[j];
                    arr1[j] = tmp;
                    swap++;
                }
            }
        }
        Arrays.sort(arr1,swap,arr1.length);
        */

        //时间复杂度 O(m + n + mlogm + m)
        //初始化数据
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        for(int num : arr1) list.add(num);
        for(int i = 0; i < arr2.length; i++) map.put(arr2[i], i);
        //自定义排序
        list.sort((x, y) -> {
            if (map.containsKey(x) || map.containsKey(y))
                return map.getOrDefault(x, 1001) - map.getOrDefault(y, 1001);
            //arr1中未在arr2中出现过的升序排序
            return x - y;
        });
        //list -> array
        for(int i = 0; i < arr1.length; i++) arr1[i] = list.get(i);

        return arr1;
    }

    /**
     * 2020-11-15 移掉K位数字
     * https://leetcode-cn.com/problems/remove-k-digits/
     * num 的长度小于 10002 且 ≥ k
     * num 不会包含任何前导零
     * @param num 给定一个以字符串表示的非负整数num
     * @param k 移除这个数中的 k 位数字
     * @return 使得剩下的数字最小
     */
    public String removeKdigits(String num, int k) {
        //方法一 暴力贪心 复杂度最差会达到 O(nk)
        /*if (num.length() == k) return "0";
        StringBuilder s = new StringBuilder(num);
        for (int i = 0; i < k; i++) {
            int idx = 0;
            //从左到右找第一个比后面大的字符
            for (int j = 1; j < s.length() && s.charAt(j) >= s.charAt(j - 1); j++) idx = j;
            //删除它
            s.deleteCharAt(idx);
            //清除前导零
            while (s.charAt(0) == '0' && s.length() > 1)  s.deleteCharAt(0);
        }
        return s.toString();*/

        //方法二 单调栈+贪心 复杂度最差会达到 O(n)
        Deque<Character> deque = new LinkedList<>();
        for (int i = 0; i < num.length(); i++) {
            char charAt = num.charAt(i);
            //如果栈顶元素大于当前字符,表示是需要丢弃的
            while (!deque.isEmpty() && k > 0 && deque.peekLast() > charAt) {
                deque.pollLast();
                k--;
            }
            deque.offerLast(charAt);
        }
        //如果上一步没有达到所需删除的字符数
        for (int i = 0; i < k; i++) deque.pollLast();

        //构造结果数字字符串
        StringBuilder rst = new StringBuilder();
        boolean leadingZero = true; //前导零判断
        while (!deque.isEmpty()) {
            char digit = deque.pollFirst();
            if (leadingZero && digit == '0') {
                continue;
            }
            leadingZero = false;
            rst.append(digit);
        }
        return rst.length() == 0 ? "0" : rst.toString();
    }

    /**
     * 2020-11-16 根据身高重建队列
     * https://leetcode-cn.com/problems/queue-reconstruction-by-height/
     * @param people 打乱顺序的一群人 每个人由一个整数对(h, k)表示 总人数少于1100人
     * @return 其中h是这个人的身高,k是排在这个人前面且身高大于或等于h的人数,编写一个算法来重建这个队列
     */
    public int[][] reconstructQueue(int[][] people) {
        if (people == null || people.length < 1) return people;

        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] p1, int[] p2) {
                //按照身高降序 K升序排序
                return p1[0] == p2[0] ? p1[1] - p2[1] : p2[0] - p1[0];
            }
        });

        List<int[]> list = new ArrayList<>();
        //K值定义为 排在这个人前面且身高大于或等于h的人数
        //因为从身高降序开始插入，此时所有人身高都大于等于h
        //因此K值即为需要插入的位置
        for (int[] i : people) {
            list.add(i[1], i);
        }
        return list.toArray(new int[list.size()][]);
    }

    /**
     * 2020-11-17 距离顺序排列矩阵单元格
     * https://leetcode-cn.com/problems/matrix-cells-in-distance-order/
     * @param R 1 <= R <= 100
     * @param C 1 <= C <= 100
     * @param r0 0 <= r0 < R
     * @param c0 0 <= c0 < C
     * @return 返回矩阵中的所有单元格的坐标,并按到(r0, c0)的距离从最小到最大的顺序排
     * 两单元格(r1, c1) 和 (r2, c2) 之间的距离是曼哈顿距离，|r1 - r2| + |c1 - c2|
     */
    public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
        //桶排序,先要求出桶的最大数量,空间复杂度O(RC),时间复杂度O(RC)
        TreeMap<Integer,List<int[]>> bucket = new TreeMap<>();
        int maxDist = Math.max(r0, R - 1 - r0) + Math.max(c0, C - 1 - c0);
        for (int i = 0; i <= maxDist; i++) {
            bucket.put(i,new ArrayList<>());
        }
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                int Manhattan_Distance = Math.abs(r0-r) + Math.abs(c0-c);
                bucket.get(Manhattan_Distance).add(new int[]{r,c});
            }
        }
        int[][] ret = new int[R * C][];
        int index = 0;
        for (int i = 0; i <= maxDist; i++) {
            for (int[] it : bucket.get(i)) {
                ret[index++] = it;
            }
        }
        return ret;

        //直接插入排序,空间复杂度O(RC),时间复杂度O(RClog(RC) + RC)
        /*
        int[][] rst = new int[R*C][2];
        int index = 0;
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                rst[index++] = new int[]{r,c};
            }
        }
        Arrays.sort(rst, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                int dist_1 = Math.abs(r0-o1[0]) + Math.abs(c0-o1[1]);
                int dist_2 = Math.abs(r0-o2[0]) + Math.abs(c0-o2[1]);
                return dist_1 - dist_2;
            }
        });

        return rst;
        */
    }

    /**
     * 2020-11-18 加油站
     * https://leetcode-cn.com/problems/gas-station/
     * @param gas 从第 i 个加油站有汽油 gas[i] 升
     * @param cost 从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升
     * @return 如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1
     * 输入数组均为非空数组，且长度相同
     * 输入数组中的元素均为非负数
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        //如果只有一个加油站那肯定可以
        if (gas.length != cost.length) return -1;
        //遍历假设每一个加油站都作为起始,看看是否能够绕一圈
        for (int i = 0; i < gas.length; i++) {
            //如果起始油站的油都不够走到下一程,没必要继续
            if (gas[i] < cost[i]) continue;

            //油箱初始油量有多少
            int g_volume = 0;
            for (int g = i;;) {
                //先在当前油站补充油量
                g_volume += gas[g];
                //减去当前油站出发到下一个油站需消耗的油量
                g_volume -= cost[g];

                //半路没油了
                if (g_volume < 0) break;
                g ++;
                g %= gas.length;
                if (g == i) break;
            }

            //绕完一圈还剩油则说明成立
            if (g_volume >= 0) return i;
        }
        //说明不能够走完一圈
        return -1;
    }

    /**
     * 2020-11-19 移动零
     * https://leetcode-cn.com/problems/move-zeroes/
     * 将所有0移动到数组的末尾
     * 同时保持非零元素的相对顺序
     * @param nums 给定一个数组
     */
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length < 2) return;

        //思路一:交换元素 O(n^2)
        /*
        int tmp;
        for (int i = 0; i < nums.length; i++) {
            //找到一个0
            if (nums[i] == 0) {
                //找到一个非0跟他交换
                for (int cur = i+1; cur < nums.length; cur++) {
                    if (nums[cur] != 0) {
                        tmp = nums[i];
                        nums[i] = nums[cur];
                        nums[cur] = tmp;
                        break;
                    } else if (cur == nums.length-1) return;
                }
            }
        }
        */

        //思路二:先处理非0 O(n)
        int current = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[current++] = nums[i];
            }
        }
        for (int i = current; i < nums.length; i++) {
            nums[current++] = 0;
        }
    }

    /**
     * 2020-11-20 对链表进行插入排序 O(n^2)
     * https://leetcode-cn.com/problems/insertion-sort-list/
     * @param head 未排序前链表表头
     * @return 排序后链表表头
     */
    public ListNode insertionSortList(ListNode head) {
        if (head == null) return null;
        //设置一个傀儡指针
        ListNode puppet = new ListNode(Integer.MIN_VALUE);

        while (head != null) {
            //找到需要插入的位置
            ListNode tmp = puppet;
            while (tmp.next != null) {
                if (head.val < tmp.next.val) break;
                tmp = tmp.next;
            }
            //提前保留下一次的头指针
            ListNode nextNext = head.next;
            //执行插入排序
            ListNode next = tmp.next;
            tmp.next = head;
            head.next = next;
            //进行下一次遍历
            head = nextNext;
        }
        return puppet.next;
    }

    /**
     * 2020-11-21 升序排序链表
     * https://leetcode-cn.com/problems/sort-list/
     * 要求O(nlogn)时间复杂度和常数级空间复杂度
     * 0 ≤ 链表中节点数目 ≤ 5*104
     * -10^5 <= Node.val <= 10^5
     * @param head 未排序前链表表头
     * @return 排序后链表表头
     */
    public ListNode sortList(ListNode head) {
        if (head == null) return null;

        int length = 0;
        ListNode tmp = head;
        while (tmp != null) {
            tmp = tmp.next;
            length++;
        }

        //自底向上归并排序
        ListNode puppet = new ListNode(Integer.MIN_VALUE);
        puppet.next = head;
        for (int step = 1; step < length; step <<= 1) {
            ListNode prev = puppet, curr = puppet.next;
            while (curr != null) {
                //划分左边
                ListNode left = curr;
                for (int i = 1; i < step && curr.next != null; i++) {
                    curr = curr.next;
                }
                ListNode right = curr.next;
                curr.next = null;
                //划分右边
                curr = right;
                for (int i = 1; i < step && curr != null && curr.next != null; i++) {
                    curr = curr.next;
                }
                ListNode next = null;
                if (curr != null) {
                    next = curr.next;
                    curr.next = null;
                }
                //归并
                prev.next = merge(left, right);
                while (prev.next != null) {
                    prev = prev.next;
                }
                curr = next;
            }
        }
        return puppet.next;
    }
    public ListNode merge(ListNode left, ListNode right) {
        ListNode dummyHead = new ListNode(Integer.MIN_VALUE);
        ListNode temp = dummyHead, temp1 = left, temp2 = right;
        //大小比对调整
        while (temp1 != null && temp2 != null) {
            if (temp1.val <= temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }
        //合并两条链表
        if (temp1 != null) {
            temp.next = temp1;
        } else if (temp2 != null) {
            temp.next = temp2;
        }
        return dummyHead.next;
    }

    /**
     * 2020-11-22 有效的字母异位词
     * https://leetcode-cn.com/problems/valid-anagram/
     * @param s 字符串 s
     * @param t 字符串 t
     * @return 判断 t 是否是 s 的字母异位词
     */
    public boolean isAnagram(String s, String t) {
        if (s == null || t == null) return false;

        //直接比较,时间复杂度O(n^2) 空间O(n)
        /*StringBuilder stb = new StringBuilder(s);
        for (int i = 0; i < t.length(); i++) {
            char cur = t.charAt(i);
            int index = stb.indexOf(String.valueOf(cur));
            if (index == -1) return false;
            stb.deleteCharAt(index);
        }
        return stb.length() == 0;*/

        //排序后比较,时间复杂度O(n) 空间O(n)
        char[] ss = s.toCharArray();
        char[] ts = t.toCharArray();
        Arrays.sort(ss);
        Arrays.sort(ts);
        return Arrays.equals(ss, ts);

        //哈希桶排序,时间复杂度O(n) 空间O(n)
        /*Map<Character,Integer> map = new HashMap<>(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            Integer count = map.get(c);
            if (count == null) return false;
            count--;
            if (count == 0) map.remove(c);
            else map.put(c,count);
        }
        return map.size() == 0;*/
    }

    /**
     * 2020-11-23 用最少数量的箭引爆气球
     * https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/
     * points[i].length == 2 表示气球的开始横坐标和结束横坐标
     * -2^31 <= points[i][0] < points[i][1] <= 2^31 - 1
     * @param points 0 <= points.length <= 10^4
     * @return 引爆所有气球所必须射出的最小弓箭数
     */
    public int findMinArrowShots(int[][] points) {
        if (points == null || points.length < 1) return 0;
        if (points.length < 2) return 1;

        //按照每个气球的右边界对数组进行排序,不能按照左边界
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] point1, int[] point2) {
                if (point1[1] > point2[1]) return 1;
                else if (point1[1] < point2[1]) return -1;
                else return 0;
            }
        });

        //默认向第一个气球射出一箭
        int pos = points[0][1];
        int minArrow = 1;
        for (int[] balloon: points) {
            //贪心判断这支箭能够射掉多少个气球
            if (balloon[0] > pos) {
                pos = balloon[1];
                ++minArrow;
            }
        }

        return minArrow;
    }

    /**
     * 2020-11-24 完全二叉树的节点个数
     * https://leetcode-cn.com/problems/count-complete-tree-nodes/
     * @param root 完全二叉树的根
     * @return 完全二叉树的节点个数
     */
    public int countNodes(TreeNode root) {
        if (root == null) return 0;
        int count = 0;

        //使用前 中 后 层序遍历
        Stack<TreeNode> stack = new Stack<>();

        /*stack.add(null);
        while (!stack.isEmpty()){
            count++;
            if (root.right!=null) stack.add(root.right);
            if (root.left!=null) root = root.left;
            else root = stack.pop();
        }*/

        while (!stack.isEmpty() || root != null){
            if (root!=null) {
                stack.add(root);
                root = root.left;
            }
            else {
                root = stack.pop();
                count++;
                root = root.right;
            }
        }

        return count;
    }

    /**
     * 2020-11-25 上升下降字符串
     * https://leetcode-cn.com/problems/increasing-decreasing-string/
     * 给你一个字符串 s ，请你根据下面的算法重新构造字符串：
     *
     * 1.从 s 中选出 最小 的字符，将它接在结果字符串的后面。
     * 2.从 s 剩余字符中选出 最小 的字符，且该字符比上一个添加的字符大，将它接在结果字符串后面。
     * 3.重复步骤 2 ，直到你没法从 s 中选择字符。
     * 4.从 s 中选出 最大 的字符，将它 接在 结果字符串的后面。
     * 5.从 s 剩余字符中选出 最大 的字符，且该字符比上一个添加的字符小，将它接在结果字符串后面。
     * 6.重复步骤 5 ，直到你没法从 s 中选择字符。
     * 7.重复步骤 1 到 6 ，直到 s 中所有字符都已经被选过。
     * 在任何一步中，如果最小或者最大字符不止一个，你可以选择其中任意一个，并将其添加到结果字符串。
     * @param s 1 <= s.length <= 500 && ‘a’ <= s[i] <= 'z'
     * @return 将s中字符重新排序后的结果字符串
     */
    public String sortString(String s) {
        if (s == null || s.length() < 2) return s;
        char[] s_chars = s.toCharArray();
        Arrays.sort(s_chars);

        StringBuilder result = new StringBuilder();

        //true表示步骤123,false表示步骤456
        boolean flag = true;
        char lastAppended = '0';

        int pivot = 0;
        //重复步骤 1 到 6 ，直到 s 中所有字符都已经被选过
        while (result.length() != s_chars.length) {
            //步骤1-2-3
            if (flag) {
                if (s_chars[pivot] > lastAppended && s_chars[pivot] != '0') {
                    result.append(s_chars[pivot]);
                    lastAppended = s_chars[pivot];
                    s_chars[pivot] = '0';
                }
                pivot++;
                if (pivot == s_chars.length) {
                    flag = false;
                    lastAppended = '{';
                    pivot--;
                }
            }
            //步骤4-5-6
            else {
                if (s_chars[pivot] < lastAppended && s_chars[pivot] != '0') {
                    result.append(s_chars[pivot]);
                    lastAppended = s_chars[pivot];
                    s_chars[pivot] = '0';
                }
                pivot--;
                if (pivot == -1) {
                    flag = true;
                    lastAppended = '0';
                    pivot++;
                }
            }
        }

        return result.toString();
    }

    /**
     * 2020-11-26 最大间距
     * https://leetcode-cn.com/problems/maximum-gap/
     * 给定一个无序的数组,找出数组在排序之后,相邻元素之间最大的差值
     * 请尝试在线性时间复杂度和空间复杂度的条件下解决此问题
     * @param nums 0 <= nums[i] <= 2^32 - 1
     * @return 最大间距
     */
    public int maximumGap(int[] nums) {
        if (nums == null || nums.length < 2) return 0;

        int maxGap = 0;
        Arrays.sort(nums);

        for (int i = 1; i < nums.length; i++) {
            int gap = nums[i] - nums[i-1];
            if (gap > maxGap) maxGap = gap;
        }
        return maxGap;
    }

    /**
     * 2020-11-27 四数相加 II
     * https://leetcode-cn.com/problems/4sum-ii/
     * 计算有多少个元组 (i, j, k, l)，使得 A[i] + B[j] + C[k] + D[l] = 0
     * 四个整数数组的长度相同
     * @param A 整数数组A  0 <= A.length <= 500
     * @param B 整数数组B  0 <= B.length <= 500
     * @param C 整数数组C  0 <= C.length <= 500
     * @param D 整数数组D  0 <= D.length <= 500
     * @return 满足条件的元组数量
     * 整数的范围在 -2^28 到 2^28 - 1 之间
     * 最终四数相加结果不会超过 2^31 - 1
     */
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        //分组+哈希建立映射关系
        //key最为AB数组中元素组合的值,value记录出现的次数
        Map<Integer,Integer> map = new HashMap<>();

        //时间复杂度O(n^2) 空间复杂度O(n^2)
        for (int a : A) {
            for (int b : B) {
                int ab = a + b;
                map.put(ab,map.getOrDefault(ab,0)+1);
            }
        }

        //再遍历CD数组，找到0-(c+d)在map中出现过的出现次数统计出来
        //时间复杂度O(n^2) 空间复杂度O(1)
        int count = 0;
        for (int c : C) {
            for (int d : D) {
                int cd = -(c + d);
                count += map.getOrDefault(cd,0);
            }
        }

        return count;
    }

    /**
     * 2020-11-28 翻转对
     * https://leetcode-cn.com/problems/reverse-pairs/
     * 给定一个数组nums,如果i < j且nums[i] > 2*nums[j]我们就将(i,j)称作一个重要翻转对
     * @param nums nums.length <= 50000 && 2^32 <= nums[i] <= 2^32 - 1
     * @return 给定数组中的重要翻转对的数量
     */
    public int reversePairs(int[] nums) {
        if (nums == null || nums.length < 2) return 0;

        // 时间O(n^2) 超时需要使用分治法
        /*int count = 0;
        for (int j = 0; j < nums.length; j++) {
            for (int i = 0; i < j; i++) {
                long nums_i = nums[i];
                long nums_j = nums[j];
                if (nums_i > (nums_j<<1)) count++;
            }
        }
        return count;*/

        //时间O(nlogn) 归并排序分治思想
        return reversePairsRecursive(nums, 0, nums.length - 1);
    }
    public int reversePairsRecursive(int[] nums, int left, int right) {
        if (left == right) {
            return 0;
        } else {
            int mid = (left + right) / 2;
            int n1 = reversePairsRecursive(nums, left, mid);
            int n2 = reversePairsRecursive(nums, mid + 1, right);
            int ret = n1 + n2;

            // 首先统计下标对的数量
            //则nums[l..r]中的翻转对数目,就等于两个子数组的翻转对数目之和
            int i = left;
            int j = mid + 1;
            while (i <= mid) {
                while (j <= right && (long) nums[i] > 2 * (long) nums[j]) {
                    j++;
                }
                ret += j - mid - 1;
                i++;
            }

            // 随后合并两个排序数组
            int[] sorted = new int[right - left + 1];
            int p1 = left, p2 = mid + 1;
            int p = 0;
            while (p1 <= mid || p2 <= right) {
                if (p1 > mid) {
                    sorted[p++] = nums[p2++];
                } else if (p2 > right) {
                    sorted[p++] = nums[p1++];
                } else {
                    if (nums[p1] < nums[p2]) {
                        sorted[p++] = nums[p1++];
                    } else {
                        sorted[p++] = nums[p2++];
                    }
                }
            }
            for (int k = 0; k < sorted.length; k++) {
                nums[left + k] = sorted[k];
            }
            return ret;
        }
    }

    /**
     * 2020-11-29 三角形的最大周长
     * https://leetcode-cn.com/problems/largest-perimeter-triangle/
     * 给定由一些正数（代表长度）组成的数组 A
     * @param A 3 <= A.length <= 10000
     * @return 返回由其中三个长度组成的,面积不为零的三角形的最大周长
     * 如果不能形成任何面积不为零的三角形，返回0
     */
    public int largestPerimeter(int[] A) {
        //首先对数组进行排序
        Arrays.sort(A);

        //逆序判断,即可找到最大周长
        for (int i = A.length-1; i >= 2; i--) {
            int a = A[i];
            int b = A[i-1];
            int c = A[i-2];
            if (a < (b+c)) return a+b+c;
        }

        return 0;
    }

    /**
     * https://leetcode-cn.com/problems/number-of-segments-in-a-string/comments/
     * 统计字符串中的每个单词个数,这里的单词指的是连续的不是空格的字符
     * 请注意,你可以假定字符串里不包括任何不可打印的字符
     * @param sentence 给定一个字符串
     * @return 每个单词个数
     */
    public Map<String,Integer> CountWord(String sentence) {
        if (sentence == null) return null;

        Map<String,Integer> map = new HashMap<>();

        for (int i = 0; i < sentence.length(); i++) {
            char s = sentence.charAt(i);
            if (('a' <= s && s <= 'z') || ('A' <= s && s <= 'Z')) {
                int j = i+1;
                for (; j < sentence.length(); j++) {
                    char e = sentence.charAt(j);
                    if (!(('a' <= e && e <= 'z') || ('A' <= e && e <= 'Z'))) break;
                }
                String word = sentence.substring(i, j);
                map.put(word,map.getOrDefault(word,0)+1);
                i = j;
            }
        }

        return map;
    }

    /**
     * 2020-11-30 重构字符串
     * https://leetcode-cn.com/problems/reorganize-string/
     * @param S 给定一个字符串S
     * @return 检查是否能重新排布其中的字母，使得两相邻的字符不同。
     * 计数+贪心
     */
    public String reorganizeString(String S) {
        if (S == null || S.length() < 2) return S;

        /*将S中所有字符及其出现次数录入数组*/
        int[] counts = new int[26];
        int length = S.length();
        int threshold = (length + 1) >> 1;
        int maxCount = 0;
        for (int i = 0; i < length; i++) {
            char c = S.charAt(i);
            counts[c - 'a']++;
            maxCount = Math.max(maxCount, counts[c - 'a']);
            /*如果最多个数的字符数量大于字符串长度+1的一半时,则其他字符依次插孔也无法分割,返回空字符串*/
            if (maxCount > threshold) return "";
        }

        /*到此说明可以重组字符串*/
        char[] reorganizeArray = new char[length];
        int evenIndex = 0, oddIndex = 1;
        int halfLength = length / 2;
        for (int i = 0; i < 26; i++) {
            char c = (char) ('a' + i);
            while (counts[i] > 0 && counts[i] <= halfLength && oddIndex < length) {
                reorganizeArray[oddIndex] = c;
                counts[i]--;
                oddIndex += 2;
            }
            while (counts[i] > 0) {
                reorganizeArray[evenIndex] = c;
                counts[i]--;
                evenIndex += 2;
            }
        }
        return new String(reorganizeArray);
    }

    /**
     * 2020-12-01 在排序数组中查找元素的第一个和最后一个位置
     * https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
     * 给定一个按照升序排列的整数数组 nums，和一个目标值 target
     * @param nums 0 <= nums.length <= 105  &&  -109 <= nums[i] <= 109
     * @param target -109 <= target <= 109
     * @return 找出给定目标值在数组中的开始位置和结束位置，如果数组中不存在目标值 target，返回 [-1, -1]
     */
    public int[] searchRange(int[] nums, int target) {
        int[] result = {-1,-1};
        if(nums.length < 1) return result;

        int left = 0, right = nums.length-1, pivot;
        //先找左边界 O(logn)
        while (left <= right) {
            pivot = (left + right) >> 1;

            if (nums[pivot] > target) right = pivot-1;
            else if (target > nums[pivot]) left = pivot+1;
            else {
                // target == nums[pivot];
                result[0] = pivot;
                right = pivot-1;
            }
        }

        left = 0;
        right = nums.length-1;
        //再找右边界 O(logn)
        while (left <= right) {
            pivot = (left + right) >> 1;

            if (nums[pivot] > target) right = pivot-1;
            else if (target > nums[pivot]) left = pivot+1;
            else {
                // target == nums[pivot];
                result[1] = pivot;
                left = pivot+1;
            }
        }

        return result;
    }

    /**
     * 2020-12-02 拼接最大数
     * https://leetcode-cn.com/problems/create-maximum-number/
     * @param nums1 0 <= nums1[i] <= 9
     * @param nums2 0 <= nums2[i] <= 9
     * @param k k <= m + n
     * @return 
     */
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {

        return null;
    }

    /**
     * 2020-12-03 计数质数
     * https://leetcode-cn.com/problems/count-primes/
     * @param n 0 <= n <= 5 * 10^6
     * @return 统计所有小于非负整数n的质数的数量
     */
    public int countPrimes(int n) {
        if (n <= 2) return 0;

        //埃氏筛选法
        //思路: 建立一个byte型数组M
        //若已知一个数M[k]是质数,那么其i(i为正整数)倍M[k*i]必然为合数,可将其去除
        byte[] origin = new byte[n+1];
        int count = 0;
        for(int i=2;i<n;i++){
            if(origin[i] == 0){
                count++;
                int k = 2;
                while(i*k <= n){
                    origin[i*k] = 1;
                    k++;
                }
            }
        }
        return count;
    }

    /**
     * 2020-12-04 分割数组为连续子序列
     * https://leetcode-cn.com/problems/split-array-into-consecutive-subsequences/
     * @param nums 按升序排序的整数数组num(可能包含重复数字)
     * @return 请你将它们分割成一个或多个子序列，其中每个子序列都由连续整数组成且长度至少为3
     */
    public boolean isPossible(int[] nums) {
        // 长度范围1到10000，不用做特殊值判断
        // 构建最小堆
        Map<Integer, PriorityQueue<Integer>> minQueMap = new HashMap<>();
        // 数组是升序的，直接遍历
        for (int num : nums) {
            // num第一次被遍历到，加入map
            if (!minQueMap.containsKey(num)) {
                minQueMap.put(num, new PriorityQueue<Integer>());
            }
            // 判断与其链接的上一个数是否存在
            if (minQueMap.containsKey(num - 1)) {
                // 不用判空，因为contain的key一定value不为null
                // 拿出上一个数所在的子序列中，最短的那个子序列长度
                int poll = minQueMap.get(num - 1).poll();
                // 如果上一个数所在的子序列，最短的子序列被拿出后，上一个数没有作为结尾数所在的子序列了的话，移出map
                if (minQueMap.get(num - 1).isEmpty()) {
                    minQueMap.remove(num - 1);
                }
                // 当前数所在子序列长度为上一个数所在子序列长度+1
                minQueMap.get(num).offer(poll + 1);
            } else {
                // 初始化当前数所在子序列长度1
                minQueMap.get(num).offer(1);
            }
        }
        // 遍历结果集
        Set<Map.Entry<Integer, PriorityQueue<Integer>>> entries = minQueMap.entrySet();
        for (Map.Entry<Integer, PriorityQueue<Integer>> entry : entries) {
            PriorityQueue<Integer> value = entry.getValue();
            // 如果有的子序列里面，最长长度小于3，返回false
            if (value.peek() < 3) {
                return false;
            }
        }
        return true;
    }

    /**
     * 2020-12-04 解码方法
     * 一条包含字母 A-Z 的消息通过以下方式进行了编码：
     * 'A' -> 1
     * 'B' -> 2
     * ...
     * 'Z' -> 26
     * https://leetcode-cn.com/problems/decode-ways/
     * @param s 1 <= s.length <= 100 && s只包含数字，可能包含前导零
     * @return 给定一个只包含数字的非空字符串，请计算解码方法的总数。
     */
    public int numDecodings(String s) {
        if(s == null) return 0;
        int len = s.length();
        if(len == 1 && s.charAt(0) == '0') return 0;
        if(len == 1) return 1;

        //动态规划状态转移方程 f(n) = f(n-1) + f(n-2)
        //一开始默认两个相邻的字符都符合条件
        int n_1 = 1;
        int n_2 = 1;

        for(int i = 0; i < len; i++){
            //记录当前字符编码可能性
            int count = 0;
            if(s.charAt(i) != '0') count += n_2;
            if(i > 0 && (s.charAt(i-1) == '1' || (s.charAt(i-1) == '2' && s.charAt(i) <= '6')))
                count += n_1;
            n_1 = n_2;
            n_2 = count;
        }

        return n_2;
    }

    /**
     * 2020-12-04 数组中出现次数超过一半的数字
     * https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/
     * @param nums 非空数组
     * @return 数组中一定有一个数字出现的次数超过数组长度的一半,请找出这个数字
     */
    public int majorityElement(int[] nums) {
        /*
        //哈希表计数 时间O(n) 空间O(n/2)
        Map<Integer,Integer> map = new TreeMap<>();
        int div = nums.length >> 1;

        for(int n : nums) {
            map.put(n,map.getOrDefault(n,0)+1);
        }

        int max = nums[0];
        for (int key : map.keySet()) {
            Integer count = map.get(key);
            if (count > div) max = key;
        }

        return max;
        */

        //摩尔投票 时间O(n) 空间O(1)
        int count = 0;
        int card = 0;
        for(int num:nums){
            if(count == 0) card = num;
            //通过不停地大战消耗，最后幸存者就是赢家
            count += (card == num)?1:-1;
        }
        return card;
    }

    /**
     * 2020-12-05 任务调度器
     * 给你一个用字符数组tasks 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。
     * 任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。
     *
     * 然而，两个相同种类的任务之间必须有长度为整数 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。
     * https://leetcode-cn.com/problems/task-scheduler/
     * @param tasks 1 <= task.length <= 10^4 && 'A' <= tasks[i] <= 'Z'
     * @param n 1 <= n <= 100
     * @return 计算完成所有任务所需的最短时间
     */
    public int leastInterval(char[] tasks, int n) {
        //步骤1 先记录下每种任务要执行的次数
        int[] taskCount = new int[26];
        for (char task : tasks) taskCount[task-'A']++;

        //步骤2 对每种任务的次数进行排序,找出次数最大任务
        Arrays.sort(taskCount);
        int maxCount = taskCount[25];
        //总排队时间 = (桶个数 - 1) * (n + 1) + 最后一桶的任务数
        int retCount = (maxCount - 1) * (n + 1) + 1;
        //步骤3 找出跟次数最大任务相同的任务
        int i = 24;
        while (i >= 0 && taskCount[i--] == maxCount) retCount++;
        //步骤4 Math.max(存在空闲时间 不存在空闲时间)
        return Math.max(retCount, tasks.length);
    }

    /**
     * 2020-12-06 杨辉三角形
     * https://leetcode-cn.com/problems/pascals-triangle/
     * @param numRows 给定一个非负整数numRows
     * @return 生成杨辉三角的前numRows行
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> lists = new ArrayList<>();
        if (numRows == 0) return lists;
        List<Integer> row_one = new ArrayList<>();
        row_one.add(1);
        lists.add(row_one);
        if (numRows == 1) return lists;

        for (int i = 1; i < numRows; i++) {
            List<Integer> prev_row = lists.get(i-1);
            List<Integer> curr_row = new ArrayList<>();
            curr_row.add(1);
            for (int j = 1; j < prev_row.size(); j++) {
                curr_row.add(prev_row.get(j-1) + prev_row.get(j));
            }
            curr_row.add(1);
            lists.add(curr_row);
        }

        return lists;
    }

    /**
     * 2020-12-07 翻转矩阵后的得分
     * 有一个二维矩阵 A 其中每个元素的值为 0 或 1
     * https://leetcode-cn.com/problems/score-after-flipping-matrix/
     * @param A 1 <= A.length <= 20 && 1 <= A[0].length <= 20
     * @return 将该矩阵的每一行都按照二进制数来解释,矩阵的得分就是这些数字的总和,返回尽可能高的分数
     */
    public int matrixScore(int[][] A) {
        //首先保证每行是最大的
        for (int[] value : A) {
            int oldAi = binary_to_decimal(value);
            binary_arr_reverse(value);
            int newAi = binary_to_decimal(value);
            if (oldAi > newAi) binary_arr_reverse(value);
        }
        //再次保证每列1的数量最多
        for (int i = 0; i < A[0].length; i++) {
            int zero = 0;
            int one = 0;
            for (int[] ints : A) {
                if (ints[i] == 0) zero++;
                else one++;
            }
            if (one < zero) {
                for (int j = 0; j < A.length; j++) A[j][i] ^= 1;
            }
        }

        int result = 0;
        for (int[] ints : A) result += binary_to_decimal(ints);
        return result;
    }
    private int binary_to_decimal(int[] binary){
        int decimal = 0;
        if (binary == null) return decimal;
        for (int i = 0; i < binary.length; i++) decimal += (binary[binary.length-i-1]) << i;
        return decimal;
    }
    private void binary_arr_reverse(int[] binary_arr){
        if (binary_arr == null) return;
        for (int i = 0; i < binary_arr.length; i++) binary_arr[i] ^= 1;
    }

    /**
     * 2020-12-08 将数组拆分成斐波那契序列
     * https://leetcode-cn.com/problems/split-array-into-fibonacci-sequence/
     * 0 <= F[i] <= 2^31 - 1，（也就是说，每个整数都符合 32 位有符号整数类型）；
     * F.length >= 3；
     * 对于所有的0 <= i < F.length - 2，都有 F[i] + F[i+1] = F[i+2] 成立
     * 将字符串拆分成小块时，每个块的数字一定不要以零开头，除非这个块是数字 0 本身
     * @param S 1 <= S.length <= 200 && 0 <= S[i] <= 9
     * @return 返回从 S 拆分出来的任意一组斐波那契式的序列块，如果不能拆分则返回 []
     */
    public List<Integer> splitIntoFibonacci(String S) {
        List<Integer> result = new ArrayList<>();
        if (S == null || S.length() < 3) return result;

        return fibonacci_back_trace(result,0,S) ? result : new ArrayList<>();
    }
    private boolean fibonacci_back_trace(List<Integer> result,int index,String S) {
        int size = result.size();
        if (index == S.length()) return size > 2;

        int num = 0;
        for (int i = index;i < S.length();i++){
            num = 10 * num + S.charAt(i) - '0';
            //int溢出判断
            if (num < 0) return false;

            if (size < 2 || num == result.get(size-1) + result.get(size-2)) {
                result.add(num);
                if (fibonacci_back_trace(result,i+1,S)) return true;
                result.remove(size);
            }

            //判断前导0
            if (S.charAt(i) == '0' && i == index) return false;
        }

        return false;
    }

    /**
     * 2020-12-09 不同路径
     * https://leetcode-cn.com/problems/unique-paths/
     * 一个机器人位于一个 m x n 网格的左上角
     * 机器人每次只能向下或者向右移动一步
     * 机器人试图达到网格的右下角
     * @param m 网格的行数 1 <= m.length <= 100
     * @param n 网格的列数 1 <= n.length <= 100
     * @return 问总共有多少条不同的路径?  题目数据保证答案小于等于2 * 10^9
     */
    public int uniquePaths(int m, int n) {
        //DFS超时
        //boolean[][] grid = new boolean[m][n];
        //detectUniquePath(grid,0,0);
        //return path;

        //动态规划
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) dp[i][j] = 1;
                else dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m-1][n-1];
    }
    int path = 0;
    private void detectUniquePath(boolean[][] grid, int x, int y) {
        if (x >= grid.length || y >= grid[0].length || grid[x][y]) return;
        if (x == grid.length-1 && y == grid[0].length-1){
            path++;
            return;
        }
        grid[x][y] = true;
        detectUniquePath(grid,x+1,y);
        detectUniquePath(grid,x,y+1);
        grid[x][y] = false;
    }

    /**
     * 2020-12-10 柠檬水找零
     * 在柠檬水摊上，每一杯柠檬水的售价为 5 美元
     * 顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯
     * 每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元
     * 你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元
     * 注意，一开始你手头没有任何零钱
     * https://leetcode-cn.com/problems/lemonade-change/
     * @param bills 0<=bills.length<=10000 && bills[i]∈[5,10,20]
     * @return 如果你能给每位顾客正确找零，返回true，否则返回false
     */
    public boolean lemonadeChange(int[] bills) {
        if (bills.length != 0 && bills[0] != 5) return false;

        int five = 0;
        int ten = 0;
        int twenty = 0;

        /*for (int bill : bills) {
            //如果给5块照收
            if (bill == 5) five++;
            //如果给10块,看看有没有足够5块找
            else if (bill == 10) {
                if (five > 0) {
                    five--;
                    ten++;
                }
                else return false;
            }
            //如果给20块,看看有没有足够5或10块找
            else {
                if ((five*5 + ten*10) < 15) return false;
                else {
                    if (ten > 0 && five > 0) {
                        ten--;
                        five--;
                    }else if (five >= 3) {
                        five-=3;
                    }else return false;
                }
            }
        }*/

        //优化判断,只要five < 0 则终止
        for (int bill : bills) {
            //如果给5块照收
            if (bill == 5) five++;
            //如果给10块先按照最优情况判断
            else if (bill == 10) {
                five--;
                ten++;
            }
            //如果给20块先按照有10块情况判断
            else if (ten > 0){
                ten--;
                five--;
            }
            //再假设有足够的5块情况判断
            else {
                five-=3;
            }

            //最后再检查有没有足够5块找
            if (five < 0) return false;
        }

        return true;
    }

    /**
     * 2020-12-11 Dota2 参议院
     * https://leetcode-cn.com/problems/dota2-senate/
     * 他们以一个基于轮为过程的投票进行。在每一轮中，每一位参议员都可以行使两项权利:
     * 1.参议员可以让另一位参议员在这一轮和随后的几轮中丧失所有的权利
     * 2.如果参议员发现有权利投票的参议员都是同一个阵营的，他可以宣布胜利并决定在游戏中的有关变化
     * @param senate 1 <= senate.length <= 10000
     * @return 预测哪一方最终会宣布胜利并在 Dota2 游戏中决定改变
     */
    public String predictPartyVictory(String senate) {
        List<Integer> Radiant_index = new LinkedList<>();
        List<Integer> Dire_index = new LinkedList<>();
        for (int i = 0; i < senate.length(); i++) {
            if (senate.charAt(i) == 'R') Radiant_index.add(i);
            else Dire_index.add(i);
        }

        while (true) {
            if (Radiant_index.size() == 0) return "Dire";
            if (Dire_index.size() == 0) return "Radiant";

            int direIndex = Dire_index.remove(0);
            int radiantIndex = Radiant_index.remove(0);

            //模拟轮投票,如果当前是Radiant作为主场,那么在Radiant_index中增加一次投票权
            if (radiantIndex < direIndex) Radiant_index.add(radiantIndex + senate.length());
            //模拟轮投票,如果当前是Dire作为主场,那么在Dire_index中增加一次投票权
            else Dire_index.add(direIndex + senate.length());
        }
    }

    /**
     * 2020-12-12 摆动序列
     * https://leetcode-cn.com/problems/wiggle-subsequence/
     * 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列
     * 第一个差（如果存在的话）可能是正数或负数
     * 少于两个元素的序列也是摆动序列
     * @param nums 给定一个整数序列
     * @return 返回作为摆动序列的最长子序列的长度。
     * 通过从原始序列中删除一些（也可以不删除）元素来获得子序列，剩下的元素保持其原始顺序。
     */
    public int wiggleMaxLength(int[] nums) {
        if(nums.length < 2) return nums.length;

        List<Integer> list = new ArrayList<>(nums.length);
        for (int num : nums) list.add(num);

        Boolean flag = null;
        Iterator<Integer> iterator = list.iterator();
        int prev = iterator.next();
        while (iterator.hasNext()) {
            int cur = iterator.next();
            if (cur == prev) iterator.remove();
            else {
                boolean b = cur - prev > 0;
                if (flag != null && flag == b) iterator.remove();
                prev = cur;
                flag = b;
            }
        }

        return list.size();
    }

    /**
     * 2020-12-13 存在重复元素
     * https://leetcode-cn.com/problems/contains-duplicate/
     * @param nums 给定一个整数数组
     * @return 如果任意一值在数组中出现至少两次，函数返回true
     */
    public boolean containsDuplicate(int[] nums) {
        if (nums == null || nums.length < 2) return false;

        //方法一: 排序后判断
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; i++) if (nums[i] == nums[i-1]) return true;

        //方法二: Set自带去重
        /*Set<Integer> set = new HashSet<>(nums.length);
        for (int num : nums) if (!set.add(num)) return true;*/

        return false;
    }

    /**
     * 2020-12-14 字母异位词分组
     * https://leetcode-cn.com/problems/group-anagrams/
     * @param strs 给定一个字符串数组,所有输入均为小写字母
     * @return 将字母异位词组合在一起
     * 字母异位词指字母相同，但排列不同的字符串
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new ArrayList<>();
        if (strs == null || strs.length == 0) return result;

        //关联字符串的字符数组Hash值相同的List下标
        Map<Integer,Integer> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            int hashCode = Arrays.hashCode(chars);
            Integer listIndex = map.get(hashCode);
            if (listIndex == null) {
                result.add(new ArrayList<>());
                listIndex = result.size() - 1;
                map.put(hashCode, listIndex);
            }
            result.get(listIndex).add(str);
        }

        return result;
    }

    /**
     * 2020-12-15 单调递增的数字
     * https://leetcode-cn.com/problems/monotone-increasing-digits/
     * @param N 0 <= N <= 10^9
     * @return 给定一个非负整数 N，找出小于或等于 N 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增
     * （当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的）
     */
    public int monotoneIncreasingDigits(int N) {
        //base case
        if (N < 10) return N;

        char[] chars = String.valueOf(N).toCharArray();
        int lastModify = chars.length;
        for (int i = chars.length-1; i > 0; i--) {
            //当前位
            char l = chars[i];
            //当前的前一位
            char p = chars[i-1];
            //如果当前位比前一位小,前一位减1,标记当前位
            if (l < p) {
                chars[i-1]--;
                lastModify = i;
            }
        }
        //从当前位往后全部改成9
        for (int i = lastModify; i < chars.length; i++) chars[i] = '9';

        return Integer.parseInt(new String(chars));
    }

    /**
     * 2020-12-16 单词规律
     * https://leetcode-cn.com/problems/word-pattern/
     * @param pattern pattern 只包含小写字母
     * @param s str 包含了由单个空格分隔的小写字母
     * @return 给定一种规律pattern和一个字符串 str，判断str是否遵循相同的规律
     * 这里的遵循指完全匹配，例如pattern里的每个字母和字符串str中的每个非空单词之间存在着双向连接的对应规律
     */
    public boolean wordPattern(String pattern, String s) {
        String[] strings = s.split(" ");
        if (strings.length != pattern.length()) return false;
        Map<Character,String> map = new HashMap<>();

        for (int i = 0; i < pattern.length(); i++) {
            char c = pattern.charAt(i);
            if (map.containsKey(c)){
                String tmp = map.get(c);
                if (!tmp.equals(strings[i])) return false;
            }
            else if (!map.containsValue(strings[i])) map.put(c,strings[i]);
            else return false;
        }

        return true;
    }

    /**
     * 2020-12-17 买卖股票的最佳时机含手续费
     * 你可以无限次地完成交易，但是你每笔交易都需要付手续费
     * 如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票
     * 这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费
     * https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
     * @param prices 0 < prices.length <= 50000 && 0 < prices[i] < 50000
     * @param fee 0 <= fee < 50000
     * @return 获得利润的最大值
     */
    public int maxProfit(int[] prices, int fee) {
        //贪心算法
        if (prices.length < 2) return 0;
        int min = prices[0];
        int profit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < min) min = prices[i];   //找到股票最低点
            else if (prices[i] > fee + min) {       //找到有收益的一天就抛出,但是这一天并不一定是最大收益,需要考虑反悔
                profit += (prices[i] - min - fee);
                min = prices[i]-fee;                //设置最小值为当天卖出后买入，表示当前交易可能会反悔?
                //这样如果后续有prices[i] >= minnum,那么就相当于把卖出的时间点延后到prices[i]
                //当然如果遇到prices[i] < minnum,那么肯定要更新minnum了,此时的交易已经固定,不反悔
            }
        }
        return profit;

        //动态规划
        /*int[][] dp = new int[prices.length][2];       //可以进行状态压缩
        //base case
        dp[0][0] = 0;   //第一天不买
        dp[0][1] = -prices[0];  //第一天买

        for (int i = 1; i < prices.length; i++) {
            //保证没买的状态: Math.max(昨天没买今天也不买的利润,昨天我买了今天卖出去扣完手续费的利润)
            dp[i][0] = Math.max(dp[i-1][0],dp[i-1][1] + prices[i] - fee);
            //保证买了的状态: Math.max(昨天我买了今天不卖,昨天我没买今天要买并且不扣手续费)
            dp[i][1] = Math.max(dp[i-1][1],dp[i-1][0] - prices[i]);
        }

        return dp[prices.length-1][0];
        */
    }

    /**
     * 2020-12-18 找不同
     * https://leetcode-cn.com/problems/find-the-difference/
     * 字符串t由字符串s随机重排,然后在随机位置添加一个字母
     * @param s 0 <= s.length <= 1000         &&        'a' <= s[i] <= 'z'
     * @param t t.length == s.length + 1      &&        'a' <= t[i] <= 'z'
     * @return 请找出在 t 中被添加的字母
     */
    public char findTheDifference(String s, String t) {
        if (s.length() == 0) return t.charAt(0);

        /*
        int[] cnt = new int[26];
        for (int i = 0; i < s.length(); ++i) {
            char ch = s.charAt(i);
            cnt[ch - 'a']++;
        }
        for (int i = 0; i < t.length(); ++i) {
            char ch = t.charAt(i);
            cnt[ch - 'a']--;
            if (cnt[ch - 'a'] < 0) {
                return ch;
            }
        }
        return ' ';*/

        /*char[] s_chars = s.toCharArray();
        char[] t_chars = t.toCharArray();
        Arrays.sort(s_chars);
        Arrays.sort(t_chars);
        for (int i = 0; i < s_chars.length; i++) {
            if (s_chars[i] != t_chars[i]) return t_chars[i];
        }
        return t_chars[t_chars.length-1];*/

        int asciiVal = 0;
        for (int i = 0; i < t.length(); ++i) asciiVal += t.charAt(i);
        for (int i = 0; i < s.length(); ++i) asciiVal -= s.charAt(i);
        return (char) asciiVal;
    }

    /**
     * 零钱兑换
     * https://leetcode-cn.com/problems/coin-change/
     * 给定不同面额的硬币 coins 和一个总金额 amount
     * @param coins 1 <= coins.length <= 12 && 1 <= coins[i] <= 2^31 - 1
     * @param amount 0 <= amount <= 10^4
     * @return 编写一个函数来计算可以凑成总金额所需的最少的硬币个数
     */
    Map<Integer,Integer> memo = new HashMap<>();
    public int coinChange(int[] coins, int amount) {
        /*//递归写法
        //带备忘录的递归树暴力破解:时间O(k*n)
        if (memo.containsKey(amount)) return memo.get(amount);

        //base case
        if (amount == 0) return 0;
        if (amount < 0) return -1;

        //递归树暴力破解:时间O(k*n^k)
        int minCoinsCount = Integer.MAX_VALUE;
        for (int coin : coins) {
            int coinsCount = coinChange(coins,amount-coin);
            if (coinsCount == -1) continue; //如果当前硬币凑不出来
            minCoinsCount = Math.min(minCoinsCount,coinsCount+1);
        }
        memo.put(amount,minCoinsCount != Integer.MAX_VALUE ? minCoinsCount : -1);

        return memo.get(amount);*/

        //迭代写法:时间O(nk)
        //base case : 1.当amount=0时,最少需要0个硬币即可  2.当amount<0时,无解返回-1
        //包括base case共有amount+1种状态
        int[] dp = new int[amount+1];
        dp[0] = 0;

        for (int i = 1; i <= amount; i++) {
            int minCoinsCount = Integer.MAX_VALUE;
            for (int coin : coins) {
                if (i-coin < 0 || dp[i-coin] == -1) continue; //如果当前零钱凑不出来
                minCoinsCount = Math.min(minCoinsCount,dp[i-coin]+1);
            }
            dp[i] = minCoinsCount != Integer.MAX_VALUE ? minCoinsCount : -1;
        }
        return dp[amount];
    }

    /**
     * 2020-12-19 旋转图像
     * https://leetcode-cn.com/problems/rotate-image/
     * @param matrix 给定一个 n×n 的二维矩阵表示一个图像
     * 将图像顺时针旋转 90 度
     */
    public void rotate(int[][] matrix) {
        int tmp = 0;

        //首先交换每一外围的横竖数据
        for (int i = 0; i < matrix.length; i++) {
            for (int n = i+1; n < matrix.length; n++) {
                tmp = matrix[i][n];
                matrix[i][n] = matrix[n][i];
                matrix[n][i] = tmp;
            }
        }
        //然后逆序竖数据即可
        for (int i = 0; i < matrix.length; i++) {
            for (int left = 0; left < matrix.length; left++) {
                int right = matrix.length-left-1;
                if (left >= right) break;
                tmp = matrix[i][left];
                matrix[i][left] = matrix[i][right];
                matrix[i][right] = tmp;
            }
        }
    }

    /**
     * 2020-12-20 去除重复字母
     * https://leetcode-cn.com/problems/remove-duplicate-letters/
     * https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters
     * @param s 'a' <= s[i] <= 'z' && 1 <= s.length <= 10^4
     * @return 请你去除字符串中重复的字母,使得每个字母只出现一次
     * 需保证返回结果的字典序最小(要求不能打乱其他字符的相对位置)
     */
    public String removeDuplicateLetters(String s) {
        /*
        //计算每个小写字母字符的出现次数
        char[] map = new char[26];
        for (int i = 0; i < s.length(); i++) map[s.charAt(i)-'a']++;

        //标记当前字符是否访问过
        boolean[] flag = new boolean[26];
        StringBuilder stb = new StringBuilder();

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            //如果当前字符还没有访问过
            if (!flag[c-'a']) {
                //模拟栈,判断栈顶元素是否大于当前字符字母
                while (stb.length() > 0 && stb.charAt(stb.length() - 1) > c) {
                    if (map[stb.charAt(stb.length() - 1) - 'a'] > 0) {
                        flag[stb.charAt(stb.length() - 1) - 'a'] = false;
                        stb.deleteCharAt(stb.length() - 1);
                    } else {
                        //如果当前字符只剩下一个,怎样都不能删
                        break;
                    }
                }
                flag[c-'a'] = true;
                stb.append(c);
            }
            map[c-'a']--;
        }

        return stb.toString();*/

        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            Character c = s.charAt(i);
            //如果当前字符字母已经访问过
            if(stack.contains(c))
                continue;
            //判断栈顶元素是否大于当前字符字母,并且此字符字母后面还有
            while(!stack.isEmpty() && stack.peek()>c && s.indexOf(stack.peek(),i)!=-1)
                stack.pop();
            stack.push(c);
        }
        char[] chars = new char[stack.size()];
        for (int i = 0; i < stack.size(); i++) chars[i] = stack.get(i);

        return new String(chars);
    }

    /**
     * 2020-12-21 使用最小花费爬楼梯
     * https://leetcode-cn.com/problems/min-cost-climbing-stairs/
     * @param cost 2 <= cost.length <= 1000 && 0 <= cost[i] <= 100
     * @return 数组的每个索引作为一个阶梯，第i个阶梯对应着一个非负数的体力花费值cost[i](索引从0开始)。
     * 每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。
     * 您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。
     */
    public int minCostClimbingStairs(int[] cost) {
        /*int[] dp = new int[cost.length];
        //base case
        dp[0] = cost[0];
        dp[1] = cost[1];

        for (int i = 2; i < cost.length; i++) {
            //重叠子问题: 走到每一阶楼梯,判断Math.min(之前走了一步+再走两步上来,之前走了两步+再走一步上来)
            dp[i] = cost[i] + Math.min(dp[i-1],dp[i-2]);
        }

        //最后判断Math.min(之前走了一步,之前走了两步)
        return Math.min(dp[dp.length-1],dp[dp.length-2]);*/

        //状态压缩版本
        //base case
        int dp_one = cost[0];
        int dp_two = cost[1];
        for (int i = 2; i < cost.length; i++) {
            //重叠子问题: 走到每一阶楼梯,判断Math.min(之前走了一步+再走两步上来,之前走了两步+再走一步上来)
            int next = cost[i] + Math.min(dp_one,dp_two);
            dp_one = dp_two;
            dp_two = next;
        }
        //最后判断Math.min(之前走了一步,之前走了两步)
        return Math.min(dp_one,dp_two);
    }

    /**
     * 2020-12-22 二叉树的锯齿形层序遍历
     * https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal
     * @param root 二叉树的根节点
     * @return 返回其节点值的锯齿形层序遍历
     * 先从左往右,再从右往左进行下一层遍历,以此类推,层与层之间交替进行
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> traversal = new LinkedList<>();
        if (root == null) return traversal;

        //zag: true,代表从左往右
        //zig: false,代表从右往左
        boolean zag_zig = false;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            List<Integer> level = new LinkedList<>();
            int count = queue.size();
            for (int i = 0; i < count; i++) {
                TreeNode node = queue.poll();
                if (zag_zig) {
                    //先加左再加右
                    level.add(node.val);
                } else {
                    //先加右再加左,即每次在链表头部插入
                    level.add(0,node.val);
                }
                if (node.left!=null) queue.offer(node.left);
                if (node.right!=null) queue.offer(node.right);
            }
            zag_zig = !zag_zig;
            traversal.add(level);
        }
        return traversal;
    }

    /**
     * 全排列
     * https://leetcode-cn.com/problems/permutations/
     * @param nums 给定一个没有重复数字的序列
     * @return 返回其所有可能的全排列
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> permutation = new LinkedList<>();
        if (nums == null) return permutation;
        boolean[] permute_flag = new boolean[nums.length];
        permute_trace_back(permutation,new LinkedList<>(),nums,permute_flag);
        return permutation;
    }
    private void permute_trace_back(List<List<Integer>> permutation,List<Integer> trace,int[] nums,boolean[] permute_flag) {
        //结束条件
        if (trace.size() == nums.length) permutation.add(new LinkedList<>(trace));

        //可选列表
        for (int i = 0; i < nums.length; i++) {
            //此路径已被选择
            if (permute_flag[i]) continue;
            //路径选择
            trace.add(nums[i]);
            permute_flag[i] = true;
            permute_trace_back(permutation,trace,nums,permute_flag);
            //路径回溯
            permute_flag[i] = false;
            trace.remove(trace.size()-1);
        }
    }

    /**
     * 2020-12-23 字符串中的第一个唯一字符
     * https://leetcode-cn.com/problems/first-unique-character-in-a-string/
     * @param s 给定一个只包含小写字母的字符串
     * @return 找到它的第一个不重复的字符并返回它的索引
     * 如果不存在,则返回-1
     */
    public int firstUniqChar(String s) {
        List<Character> list = new ArrayList<>();
        int[] char_count = new int[26];
        for (int i = 0; i < s.length(); i++) char_count[s.charAt(i)-'a']++;

        for (int i = 0; i < s.length(); i++) {
            if (char_count[s.charAt(i)-'a'] == 1) return i;
        }
        return -1;
    }

    /**
     * 2020-12-24 分发糖果
     * https://leetcode-cn.com/problems/candy/
     * 每个孩子至少分配到 1 个糖果
     * 相邻的孩子中,评分高的孩子必须获得更多的糖果
     * @param ratings N 个孩子站成了一条直线,老师会根据每个孩子的表现预先给他们评分
     * @return 老师至少需要准备多少颗糖果
     */
    public int candy(int[] ratings){
        if (ratings == null || ratings.length == 0) return 0;
        if (ratings.length == 1) return 1;

        int[] left = new int[ratings.length];
        //先从左向右遍历: 如果当前孩子比他左边孩子评分高 则比左边孩子获得糖数量多一个糖
        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i-1]) left[i] = left[i-1]+1;
        }

        int[] right = new int[ratings.length];
        //再从右向左遍历: 如果当前孩子比他右边孩子评分高 则比右边孩子获得糖数量多一个糖
        for (int i = ratings.length-2; i > -1; i--) {
            if (ratings[i] > ratings[i+1]) right[i] = right[i+1]+1;
        }

        //最后取满足左右规则的最大糖果数
        int candys = 0;
        for (int i = 0; i < ratings.length; i++) {
            candys += Math.max(left[i],right[i]);
        }
        //一定要确保每个孩子获得一个糖果
        return candys + ratings.length;
    }

    /**
     * 2020-12-25 分发饼干
     * https://leetcode-cn.com/problems/assign-cookies/
     * 假设你是一位很棒的家长,想要给你的孩子们一些小饼干
     * 但是,每个孩子最多只能给一块饼干
     * 对每个孩子 i，都有一个胃口值g[i]，这是能让孩子们满足胃口的饼干的最小尺寸;
     * 并且每块饼干 j，都有一个尺寸 s[j]。如果 s[j]>= g[i]，我们可以将这个饼干j分配给孩子i,这个孩子会得到满足.
     * @param g 1 <= g.length <= 3 * 10^4 && 1 <= g[i] <= 2^31 - 1
     * @param s 0 <= s.length <= 3 * 10^4 && 1 <= s[j] <= 2^31 - 1
     * @return 你的目标是尽可能满足越多数量的孩子,并输出这个最大数值.
     */
    public int findContentChildren(int[] g, int[] s) {
        if (s.length == 0) return 0;
        Arrays.sort(g);
        Arrays.sort(s);
        //说明最大饼干都无法满足最小胃口(两者无交集)
        if (s[s.length-1] < g[0]) return 0;

        //找胃口和饼干尺寸的最小交集
        int g_idx = 0;
        int s_idx = 0;
        while (g_idx < g.length && s_idx < s.length) {
            if (s[s_idx] >= g[g_idx]) g_idx++;
            s_idx++;
        }
        return g_idx;
    }

    /**
     * 2020-12-26 最大矩形
     * https://leetcode-cn.com/problems/maximal-rectangle/
     * 给定一个仅包含 0 和 1
     * 大小为rows * cols的二维二进制矩阵
     * rows == matrix.length
     * cols == matrix[0].length
     * 0 <= row, cols <= 200
     * @param matrix matrix[i][j] 为 '0' 或 '1'
     * @return 找出只包含1的最大矩形,并返回其面积
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0) return 0;

        int maxArea = 0;
        int cols = matrix[0].length;
        int[] leftLessMin = new int[cols];
        int[] rightLessMin = new int[cols];
        Arrays.fill(leftLessMin, -1); //初始化为 -1，也就是最左边
        Arrays.fill(rightLessMin, cols); //初始化为 cols，也就是最右边
        int[] heights = new int[cols];
        for (int row = 0; row < matrix.length; row++) {
            //更新所有高度
            for (int col = 0; col < cols; col++) {
                if (matrix[row][col] == '1') {
                    heights[col] += 1;
                } else {
                    heights[col] = 0;
                }
            }
            //更新所有leftLessMin
            int boundary = -1; //记录上次出现 0 的位置
            for (int col = 0; col < cols; col++) {
                if (matrix[row][col] == '1') {
                    //和上次出现 0 的位置比较
                    leftLessMin[col] = Math.max(leftLessMin[col], boundary);
                } else {
                    //当前是 0 代表当前高度是 0，所以初始化为 -1，防止对下次循环的影响
                    leftLessMin[col] = -1;
                    //更新 0 的位置
                    boundary = col;
                }
            }
            //右边同理
            boundary = cols;
            for (int col = cols - 1; col >= 0; col--) {
                if (matrix[row][col] == '1') {
                    rightLessMin[col] = Math.min(rightLessMin[col], boundary);
                } else {
                    rightLessMin[col] = cols;
                    boundary = col;
                }
            }

            //更新所有面积
            for (int col = cols - 1; col >= 0; col--) {
                int area = (rightLessMin[col] - leftLessMin[col] - 1) * heights[col];
                maxArea = Math.max(area, maxArea);
            }

        }
        return maxArea;
    }

    /**
     * 全排列II
     * https://leetcode-cn.com/problems/permutations-ii/
     * @param nums 给定一个可包含重复数字的序列
     * @return 按任意顺序返回所有不重复的全排列
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> permutation = new LinkedList<>();
        if (nums == null) return permutation;
        Arrays.sort(nums);      //进行排列方便去重剪枝
        boolean[] permute_flag = new boolean[nums.length];
        permute_trace_back_unique(permutation,new LinkedList<>(),nums,permute_flag);
        return permutation;
    }
    private void permute_trace_back_unique(List<List<Integer>> permutation,List<Integer> trace,int[] nums,boolean[] permute_flag) {
        //结束条件
        if (trace.size() == nums.length) {
            permutation.add(new LinkedList<>(trace));
            return;
        }

        //可选列表
        for (int i = 0; i < nums.length; i++) {
            //此路径已被选择
            if (permute_flag[i]) continue;
            //判断当前选择是否之前已被选过
            if (i > 0 && nums[i] == nums[i-1] && !permute_flag[i - 1]) continue;
            //路径选择
            trace.add(nums[i]);
            permute_flag[i] = true;
            permute_trace_back_unique(permutation,trace,nums,permute_flag);
            //路径回溯
            permute_flag[i] = false;
            trace.remove(trace.size()-1);
        }
    }

    /**
     * 2020-12-27 同构字符串
     * https://leetcode-cn.com/problems/isomorphic-strings/
     * @param s s.length == t.length
     * @param t s.length == t.length
     * @return 给定两个字符串 s 和 t，判断它们是否是同构的
     */
    public boolean isIsomorphic(String s, String t) {
        /*
        Character[] mapping_st = new Character[128];
        Character[] mapping_ts = new Character[128];
        for (int i = 0; i < s.length(); i++) {
            char sc = s.charAt(i);
            char tc = t.charAt(i);

            if (mapping_ts[tc] != null && mapping_ts[tc] != sc) return false;
            if (mapping_st[sc] != null && mapping_st[sc] != tc) return false;

            mapping_st[sc] = tc;
            mapping_ts[tc] = sc;
        }
        return true;*/

        int[] mapping_st = new int[128];
        int[] mapping_ts = new int[128];
        int length = s.length();
        for (int i = 0; i < length; i++) {
            if (mapping_st[s.charAt(i)] != mapping_ts[t.charAt(i)]) return false;
            mapping_st[s.charAt(i)] = i+1;
            mapping_ts[t.charAt(i)] = i+1;
        }
        return true;
    }

    /**
     * 2020-12-28 买卖股票的最佳时机 IV
     * 定一个整数数组 prices，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格
     * 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）
     * https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/
     * @param k 0 <= k <= 10^9
     * @param prices 0 <= prices.length <= 1000 && 0 <= prices[i] <= 1000
     * @return 设计一个算法来计算你所能获取的最大利润,你最多可以完成k笔交易
     */
    public int maxProfit4(int k, int[] prices) {
        if (prices.length < 2) return 0;
        int profit = 0;

        //动态规划: dp[i][j][0]表示第i天交易了j次不持有股票状态, dp[i][j][1]表示第i天交易了j次持有股票状态
        int[][][] dp = new int[prices.length][k+1][2];
        //base case
        for (int i = 0; i < k+1; i++) {
            dp[0][i][0] = 0;
            dp[0][i][1] = -prices[0];
        }

        for (int i = 1; i < prices.length; i++) {
            //首先不进行交易状态转移方程,需要保证 持有|不持有 两种状态
            dp[i][0][0] = dp[i-1][0][0];
            dp[i][0][1] = Math.max(dp[i-1][0][1],-prices[i]);
            //进行j次交易的状态转移方程,需要保证 持有|不持有 两种状态
            for (int j = 1; j < dp[i].length; j++) {
                //如果今天没有股票，那我可能是昨天就没有股票，而且今天不买新的；或者昨天手里有股票，而且今天把它卖了，这样今天比昨天多完成了一笔交易
                dp[i][j][0] = Math.max(dp[i-1][j][0],dp[i-1][j-1][1]+prices[i]);
                //如果今天有股票，那我可能是昨天已经有股票了，而且今天我不卖出；或者昨天手里没有股票，而且今天我买入它。
                dp[i][j][1] = Math.max(dp[i-1][j][1],dp[i-1][j][0]-prices[i]);
            }
        }
        for (int j = 0; j < dp[0].length; j++) {
            profit = Math.max(profit,dp[prices.length-1][j][0]);
        }

        return profit;
    }

    /**
     * 2020-12-29 按要求补齐数组
     * https://leetcode-cn.com/problems/patching-array/
     * @param nums
     * @param n
     * @return
     */
    public int minPatches(int[] nums, int n) {


        return 0;
    }

    /**
     * 2020-12-30 最后一块石头的重量
     * https://leetcode-cn.com/problems/last-stone-weight/
     * 有一堆石头，每块石头的重量都是正整数。
     * 每一回合，从中选出两块 最重的 石头，然后将它们一起粉碎。
     * 假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：
     *  1.如果 x == y，那么两块石头都会被完全粉碎；
     *  2.如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
     * @param stones 1 <= stones.length <= 30 && 1 <= stones[i] <= 1000
     * @return 最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0
     */
    public int lastStoneWeight(int[] stones) {
        //术一个优先队列
        Queue<Integer> queue = new PriorityQueue<>(stones.length, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });
        for (int stone : stones) {
            queue.offer(stone);
        }

        while (queue.size() > 1) {
            int x = queue.poll();
            int y = queue.poll();
            //利用大根堆性质,这里不可能出现 x < y 情况
            if (x > y)
                queue.offer(x - y);
        }

        return queue.size() == 1 ? queue.peek() : 0;
    }

    /**
     * 2020-12-31 无重叠区间
     * https://leetcode-cn.com/problems/non-overlapping-intervals/
     * @param intervals 给定一个区间的集合
     * @return 找到需要移除区间的最小数量，使剩余区间互不重叠
     * 可以认为区间的终点总是大于它的起点。
     * 区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。
     */
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        //先对所有的区间按照结束进行升序排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }
        });

        int eraseOverlap = 0;
        for (int i = 1; i < intervals.length; i++) {
            int[] interval = intervals[0];
            //如果某一区间跟初始区间有重叠,清除它
            if (intervals[i][0] < interval[1] && intervals[i][1] > interval[0]) eraseOverlap++;
            //如果两个区间没有重叠,扩大初始区间
            else interval[1] = intervals[i][1];
        }
        return eraseOverlap;
    }

    /**
     * 2021-01-01 种花问题
     * https://leetcode-cn.com/problems/can-place-flowers/
     * 假设有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
     * @param flowerbed 1 <= flowerbed.length <= 2 * 10^4 && flowerbed[i] 为 0 或 1 && flowerbed 中不存在相邻的两朵花
     * @param n 0 <= n <= flowerbed.length
     * @return 给你一个整数数组flowerbed表示花坛，由若干0和1组成，其中0表示没种植花，1表示种植了花。
     * 另有一个数n，能否在不打破种植规则的情况下种入n朵花？
     * 能则返回 true，不能则返回 false
     */
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        //防御式编程思想
        int[] m = new int[flowerbed.length+2];
        System.arraycopy(flowerbed, 0, m, 1, m.length - 2);

        for (int i = 1; i < m.length-1; i++) {
            if (m[i] == 0 && m[i-1] == 0 && m[i+1] == 0) {
                m[i] = 1;
                n--;
            }
        }

        return n <= 0;
    }

    /**
     * 2021-01-02 滑动窗口最大值
     * https://leetcode-cn.com/problems/sliding-window-maximum/
     * 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。
     * 你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
     * @param nums 1 <= nums.length <= 10^5 && -10^4 <= nums[i] <= 10^4
     * @param k 1 <= k <= nums.length
     * @return 返回滑动窗口中的最大值。
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        //如果滑动窗口一次就能滑动完
        if (nums.length == k) {
            Arrays.sort(nums);
            return new int[]{nums[nums.length-1]};
        }

        //计算出需要滑动多少次,创建出存储集合
        int[] ret = new int[nums.length-k+1];

        //超出时间限制,时间复杂度O((n-k+1)*k) = O((n+1)k - k^2)
        /*for (int i = 0; i < ret.length; i++) {
            int max = nums[i];
            for (int j = i+1; j < i+k; j++) {
                max = Math.max(max,nums[j]);
            }
            ret[i] = max;
        }*/

        //贪心思想: 找出(下一个)大于等于(当前最大)的数字
        //双向队列 保存当前窗口最大值的数组位置 保证队列中数组位置的数按从大到小排序
        LinkedList<Integer> list = new LinkedList<>();
        for(int i=0;i<nums.length;i++) {
            // 保证从大到小 如果前面数小 弹出
            while(!list.isEmpty() && nums[list.peekLast()]<=nums[i]) {
                list.pollLast();
            }
            // 添加当前值对应的数组下标
            list.addLast(i);
            // 当滑动窗口长度为k时 下次移动先删除过期数值
            if(list.peekFirst() <= i-k) {
                list.removeFirst();
            }
            // 当滑动窗口长度为k时 再保存当前窗口中最大值
            if(i-k+1 >= 0){
                ret[i-k+1] = nums[list.peekFirst()];
            }
        }

        return ret;
    }

    /**
     * 2021-01-03 分隔链表
     * https://leetcode-cn.com/problems/partition-list/
     * @param head 链表表头
     * @param x 特定值
     * @return ListNode 给你一个链表和一个特定值 x，请你对链表进行分隔，使得所有小于 x 的节点都出现在大于或等于 x 的节点之前。
     * 你应当保留两个分区中每个节点的初始相对位置
     */
    public ListNode partition(ListNode head, int x) {
        //设置一个傀儡指针
        ListNode puppet = new ListNode(Integer.MAX_VALUE);
        puppet.next = head;

        //保存比 x 小的节点的最后位置
        ListNode small_cur = puppet;
        //保存当前游标的前一个节点
        ListNode prev = puppet;
        //保存当前游标节点
        ListNode cur = puppet;
        while (cur!=null) {
            //如果当前节点比x小
            if (cur.val < x) {
                prev.next = cur.next;
                cur.next = small_cur.next;
                small_cur.next = cur;
                small_cur = small_cur.next;
            }
            prev = cur;
            cur = cur.next;
        }

        return puppet.next;
    }

    /**
     * 2021-01-04 斐波那契数
     * https://leetcode-cn.com/problems/fibonacci-number/
     */
    public int fib(int N) {
        //时间复杂度O(2^N),空间复杂度O(1)
        /*if(N <= 1) return N;
        return fib(N-1)+fib(N-2);*/

        //时间复杂度O(N),空间复杂度O(1)
        if(N <= 1) return N;
        int first = 0,second = 1;
        for(int i = 0;i < N-1;i++) {
            int tmp = first+second;
            first = second;
            second = tmp;
        }
        return second;
    }

    /**
     * 2021-01-05 较大分组的位置
     * https://leetcode-cn.com/problems/positions-of-large-groups/
     * 我们称所有包含大于或等于三个连续字符的分组为较大分组
     * @param s 1 <= s.length <= 1000 && a <= s[i] <= a
     * @return 找到每一个较大分组的区间，按起始位置下标递增顺序排序后，返回结果
     */
    public List<List<Integer>> largeGroupPositions(String s) {
        List<List<Integer>> ret = new LinkedList<>();

        int start = 0;
        int start_count = 0;
        for (int end = 0; end < s.length(); end++) {
            if (end > 0 && s.charAt(end) == s.charAt(end-1)){
                start_count++;
            }
            else {
                if (start_count >= 3) {
                    List<Integer> group = new LinkedList<>();
                    group.add(start);
                    group.add(end-1);
                    ret.add(group);
                }
                start = end;
                start_count = 1;
            }
        }

        //防止遗漏最后一个较大分组
        if (start_count >= 3) {
            List<Integer> group = new LinkedList<>();
            group.add(start);
            group.add(s.length()-1);
            ret.add(group);
        }
        return ret;
    }

    /**
     * 2021-01-06 除法求值
     * https://leetcode-cn.com/problems/evaluate-division/
     * 给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i]
     * 另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案
     * @param equations 1 <= equations.length <= 20 && equations[i].length == 2
     *                  && 1 <= equations[i][0].length && equations[i][1].length <= 5
     *                  && equations[i][0]与equations[i][1]是由小写英文字母与数字组成的字符串
     * @param values values.length == equations.length && 0.0 < values[i] <= 20.0
     * @param queries 1 <= queries.length <= 20 && queries[i].length == 2
     *                && 1 <= queries[i][0].length && queries[i][1].length <= 5
     *                && queries[i][0]queries[i][1]是由小写英文字母与数字组成的字符串
     * @return 返回所有问题的答案,如果存在某个无法确定的答案,则用 -1.0 替代这个答案
     *         注意：输入总是有效的,你可以假设除法运算中不会出现除数为 0 的情况,且不存在任何矛盾的结果
     */
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int count=0;
        //统计出现的所有字符，并赋予对应的index
        Map<String,Integer> map=new HashMap<>();
        for (List<String> list:equations){
            for (String s:list){
                if(!map.containsKey(s)){
                    map.put(s,count++);
                }
            }
        }

        //构建一个矩阵来代替图结构
        double[][] graph=new double[count+1][count+1];

        //初始化
        for (String s:map.keySet()){
            int x=map.get(s);
            graph[x][x]=1.0;
        }
        int index=0;
        for (List<String> list:equations){
            String a=list.get(0);
            String b=list.get(1);
            int aa=map.get(a);
            int bb=map.get(b);
            double value=values[index++];
            graph[aa][bb]=value;
            graph[bb][aa]=1/value;
        }

        //通过Floyd算法进行运算
        int n=count+1;
        for (int i=0;i<n;i++){
            for (int j=0;j<n;j++){
                for (int k=0;k<n;k++){
                    if(j==k||graph[j][k]!=0) continue;
                    if(graph[j][i]!=0&&graph[i][k]!=0){
                        graph[j][k]=graph[j][i]*graph[i][k];
                    }
                }
            }
        }

        //直接通过查询矩阵得到答案
        double[] res=new double[queries.size()];
        for (int i=0;i<res.length;i++){
            List<String> q=queries.get(i);
            String a=q.get(0);
            String b=q.get(1);
            if(map.containsKey(a)&&map.containsKey(b)){
                double ans=graph[map.get(a)][map.get(b)];
                res[i]=(ans==0?-1.0:ans);
            }else {
                res[i]=-1.0;
            }
        }
        return res;
    }

    /**
     * 2021-01-07 省份数量
     * https://leetcode-cn.com/problems/number-of-provinces/
     * 有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。
     * 省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。
     * 给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。
     * @param isConnected 1 <= n <= 200 && n == isConnected.length
     *                    && n == isConnected[i].length
     *                    && isConnected[i][j] 为 1 或 0
     *                    && isConnected[i][i] == 1
     *                    && isConnected[i][j] == isConnected[j][i]
     * @return 返回矩阵中 省份 的数量
     */
    public int findCircleNum(int[][] isConnected) {
        //转换思路: 查看有多少个之间没有关联的整体
        boolean[] isVisited = new boolean[isConnected.length];
        int city_count = 0;

        //BFS:广度优先
        /*Queue<Integer> citys = new LinkedList<>();
        for (int i = 0; i < isConnected.length; i++) {
            //如果当前城市访问过，则跳过
            if (isVisited[i]) continue;
            //如果当前城市没访问过
            city_count++;
            citys.add(i);
            while (!citys.isEmpty()) {
                Integer j = citys.poll();
                isVisited[j] = true;
                for (int k = 0; k < isConnected.length; k++) {
                    if (isConnected[j][k] == 1 && !isVisited[k]) {
                        citys.add(k);
                    }
                }
            }
        }*/

        //DFS:深度优先
        for (int i = 0; i < isConnected.length; i++) {
            //如果当前城市访问过，则跳过
            if (isVisited[i]) continue;
            //如果当前城市没访问过
            dfsFindCircle(isConnected,isVisited,i);
            city_count++;
        }

        return city_count;
    }
    private void dfsFindCircle(int[][] isConnected, boolean[] isVisited, int provinceId) {
        for (int k = 0; k < isConnected.length; k++) {
            if (isConnected[provinceId][k] == 1 && !isVisited[k]) {
                isVisited[k] = true;
                dfsFindCircle(isConnected,isVisited,k);
            }
        }
    }

    /**
     * 2021-01-08 旋转数组
     * https://leetcode-cn.com/problems/rotate-array/
     * @param nums 给定一个数组
     * @param k 将数组中的元素向右移动 k 个位置，其中 k 是非负数
     */
    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 1) return;

        //时间复杂度O(nk)
        /*while (k > 0) {
            int lastOne = nums[nums.length-1];
            System.arraycopy(nums, 0, nums, 1, nums.length - 1);
            nums[0] = lastOne;
            k--;
        }*/

        //时间复杂度O(n)
        int n = nums.length;
        k %= n;
        // 第一次交换完毕后，前 k 位数字位置正确，后 n-k 位数字中最后 k 位数字顺序错误，继续交换
        for (int start = 0; start < nums.length && k != 0; n -= k, start += k, k %= n) {
            for (int i = 0; i < k; i++) {
                int tmp = nums[start + i];
                nums[start + i] = nums[nums.length - k + i];
                nums[nums.length - k + i] = tmp;
            }
        }
    }

    /**
     * 2021-01-09 买卖股票的最佳时机 III
     * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     * 设计一个算法来计算你所能获取的最大利润。
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     * https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/
     * @param prices 1 <= prices.length <= 10^5 && 0 <= prices[i] <= 10^5
     * @return 最大收益，你最多可以完成两笔交易。
     */
    public int maxProfit3(int[] prices) {
        if (prices.length < 2) return 0;
        int profit = 0;

        //暴力破解: 最大收益 = Math.max(当前最大收益,第[0-i)天之间的最大收益+第[i-price.length)天之间的最大收益) 时间O(n^2)
        /*int min,profit1,profit2;
        for (int i = 0; i < prices.length; i++) {
            min = prices[0];
            profit1 = 0;
            for (int j = 0; j < i; j++) {
                min = Math.min(min,prices[j]);
                profit1 = Math.max(profit1,prices[j]-min);
            }

            min = prices[i];
            profit2 = 0;
            for (int j = i; j < prices.length; j++) {
                min = Math.min(min,prices[j]);
                profit2 = Math.max(profit2,prices[j]-min);
            }

            profit = Math.max(profit,profit1+profit2);
        }*/

        //动态规划: dp[i][j][0]表示第i天交易了j次不持有股票状态, dp[i][j][1]表示第i天交易了j次持有股票状态
        int[][][] dp = new int[prices.length][3][2];
        //base case
        for (int i = 0; i < 3; i++) {
            dp[0][i][0] = 0;
            dp[0][i][1] = -prices[0];
        }

        for (int i = 1; i < prices.length; i++) {
            //首先不进行交易状态转移方程,需要保证 持有|不持有 两种状态
            dp[i][0][0] = dp[i-1][0][0];
            dp[i][0][1] = Math.max(dp[i-1][0][1],-prices[i]);
            //进行j次交易的状态转移方程,需要保证 持有|不持有 两种状态
            for (int j = 1; j < dp[i].length; j++) {
                //如果今天没有股票，那我可能是昨天就没有股票，而且今天不买新的；或者昨天手里有股票，而且今天把它卖了，这样今天比昨天多完成了一笔交易
                dp[i][j][0] = Math.max(dp[i-1][j][0],dp[i-1][j-1][1]+prices[i]);
                //如果今天有股票，那我可能是昨天已经有股票了，而且今天我不卖出；或者昨天手里没有股票，而且今天我买入它。
                dp[i][j][1] = Math.max(dp[i-1][j][1],dp[i-1][j][0]-prices[i]);
            }
        }
        for (int j = 0; j < dp[0].length; j++) {
            profit = Math.max(profit,dp[prices.length-1][j][0]);
        }

        return profit;
    }

    /**
     * 买卖股票的最佳时机
     * 设计一个算法来计算你所能获取的最大利润。
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     * https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
     * @param prices 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     * @return 最大收益，你最多可以完成一笔交易。
     */
    public int maxProfit(int[] prices) {
        if (prices.length < 2) return 0;
        int profit = 0;

        //暴力破解: 找到最大差额 时间O(n^2)
        /*for (int i = 0; i < prices.length; i++) {
            for (int j = i+1; j < prices.length; j++) {
                profit = Math.max(prices[j]-prices[i],profit);
            }
        }*/

        int min = Integer.MAX_VALUE;

        //贪心算法: 先想办法找到最低点,再找到最大差额 时间O(n)
        /*for (int price : prices) {
            if (price < min) min = price;
            else if (price - min > profit) profit = price - min;
        }*/

        //动态规划: 最大收益 = Math.max(前i-1天的最大收益,当天股票价格-前i-1天的最低点) 时间O(n)
        for (int price : prices) {
            min = Math.min(min,price);
            profit = Math.max(profit,price-min);
        }

        return profit;
    }

    /**
     * 2021-01-10 汇总区间
     * https://leetcode-cn.com/problems/summary-ranges/
     * 给定一个无重复元素的升序整数数组 nums
     * 返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x
     * 列表中的每个区间范围 [a,b] 应该按如下格式输出：
     *      "a->b" ，如果 a != b
     *      "a" ，如果 a == b
     * @param nums 0 <= nums.length <= 20 && -2^31 <= nums[i] <= 2^31 - 1
     * @return 区间范围列表
     */
    public List<String> summaryRanges(int[] nums) {
        List<String> ret = new LinkedList<>();

        if (nums == null || nums.length < 1) return ret;

        int start = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] - nums[start] != i-start) {
                if (i-start > 1) ret.add(nums[start] + "->" + nums[i-1]);
                else ret.add(String.valueOf(nums[start]));
                start = i;
            }
        }
        if (nums.length-start > 1) ret.add(nums[start] + "->" + nums[nums.length-1]);
        else ret.add(String.valueOf(nums[start]));

        return ret;
    }

    /**
     * 2021-01-11 交换字符串中的元素
     * https://leetcode-cn.com/problems/smallest-string-with-swaps/
     * 给你一个字符串s，以及该字符串中的一些「索引对」数组pairs，其中pairs[i] =[a, b]表示字符串中的两个索引（编号从 0 开始）。
     * 你可以任意多次交换 在pairs中任意一对索引处的字符。
     * @param s 1 <= s.length <= 10^5 && s 中只含有小写英文字母
     * @param pairs 0 <= pairs.length <= 10^5 && 0 <= pairs[i][0], pairs[i][1] < s.length
     * @return 返回在经过若干次交换后，s可以变成的按字典序最小的字符串。
     */
    public String smallestStringWithSwaps(String s, List<List<Integer>> pairs) {
        //问题分析: 交换关系具有传递性,例如: pairs = [[0, 3], [1, 2], [0, 2]] , 则 0-1-2-3 之间可以进行任意交换
        //解决方案: 找出哪些索引连在一起,把连在一起的索引按照字符的ASCII值原地升序排序

        // 第一步: 先把所有的点和可交换关系抽象成一个图
        int n = s.length();
        List<Integer>[] adjacentList = new List[n];
        for (int i = 0; i < n; i++) {
            adjacentList[i] = new ArrayList<>();
        }
        for (List<Integer> pair: pairs) {
            int a = pair.get(0);
            int b = pair.get(1);
            adjacentList[a].add(b);
            adjacentList[b].add(a);
        }

        // 第二步: DFS找到所有的连通的pairs,将对应s中字符的ASCII值升序排序
        boolean[] visited = new boolean[n];
        char[] arr = new char[n];
        for (int i = 0; i < n; i++) {
            if (visited[i]) continue;
            //存放字符串s中相互连通的字符索引
            List<Integer> pair_char_idx = new LinkedList<>();
            //递归寻找相互连通的pair
            find_connected_pair(adjacentList,i,visited,pair_char_idx);
            //将相互连通的字符进行排序后填入arr相应位置
            Queue<Character> pq = new PriorityQueue<>(Character::compareTo);
            for (Integer idx : pair_char_idx) {
                pq.offer(s.charAt(idx));
            }
            pair_char_idx.sort(Integer::compareTo);
            for (Integer idx : pair_char_idx) {
                arr[idx] = pq.poll();
            }
        }

        return new String(arr);
    }
    private void find_connected_pair(List<Integer>[] graph, int graph_node_idx, boolean[] visited, List<Integer> pair_char_idx) {
        visited[graph_node_idx] = true;
        pair_char_idx.add(graph_node_idx);
        for (int node_idx: graph[graph_node_idx]) {
            if (! visited[node_idx]) {
                find_connected_pair(graph, node_idx, visited, pair_char_idx);
            }
        }
    }

    /**
     * 2021-01-12 项目管理
     * https://leetcode-cn.com/problems/sort-items-by-groups-respecting-dependencies/
     * 公司共有 n 个项目和  m 个小组，每个项目要不无人接手，要不就由 m 个小组之一负责。
     * @param n 1 <= m <= n <= 3 * 10^4
     * @param m 1 <= m <= n <= 3 * 10^4
     * @param group group.length == beforeItems.length == n && -1 <= group[i] <= m - 1
     * @param beforeItems group.length == beforeItems.length == n
     *                    && 0 <= beforeItems[i].length <= n - 1
     *                    && 0 <= beforeItems[i][j] <= n - 1
     *                    && i != beforeItems[i][j]
     *                    && beforeItems[i] 不含重复元素
     * @return group[i] 表示第i个项目所属的小组，如果这个项目目前无人接手，那么group[i] 就等于-1。
     * （项目和小组都是从零开始编号的）小组可能存在没有接手任何项目的情况。
     * 请你帮忙按要求安排这些项目的进度，并返回排序后的项目列表：
     *  ~ 同一小组的项目，排序后在列表中彼此相邻。
     *  ~ 项目之间存在一定的依赖关系，我们用一个列表 beforeItems来表示，
     *      其中beforeItems[i]表示在进行第i个项目前（位于第 i个项目左侧）应该完成的所有项目。
     * 如果存在多个解决方案，只需要返回其中任意一个即可。如果没有合适的解决方案，就请返回一个 空列表 。
     */
    public int[] sortItems(int n, int m, int[] group, List<List<Integer>> beforeItems) {
        return null;
    }

    /**
     * 2021-01-13 冗余连接
     * 在本问题中, 树指的是一个连通且无环的无向图。
     * 输入一个图，该图由一个有着N个节点 (节点值不重复1, 2, ..., N) 的树及一条附加的边构成。附加的边的两个顶点包含在1到N中间，这条附加的边不属于树中已存在的边。
     * 结果图是一个以边组成的二维数组。每一个边的元素是一对[u, v]，满足u < v，表示连接顶点u和v的无向图的边。
     * https://leetcode-cn.com/problems/redundant-connection/
     * @param edges 输入的二维数组大小在 3 到 1000 && 二维数组中的整数在1到N之间，其中N是输入数组的大小
     * @return 返回一条可以删去的边，使得结果图是一个有着N个节点的树。
     * 如果有多个答案，则返回二维数组中最后出现的边。
     * 答案边[u, v] 应满足相同的格式u < v。
     */
    public int[] findRedundantConnection(int[][] edges) {
        //解决方案: 进行拓扑排序,找出哪些树有环,找到最后一个出现环的地方

        // 第一步: 构建邻接表和入度表
        int n = edges.length;
        int[] degrees = new int[n];
        Set<Integer>[] adjacentList = new Set[n];
        for (int i = 0; i < adjacentList.length; i++) {
            adjacentList[i] = new HashSet<>();
        }
        for(int[] edge: edges) {
            int u = edge[0]-1;
            int v = edge[1]-1;

            adjacentList[u].add(v);
            degrees[v]++;

            adjacentList[v].add(u);
            degrees[u]++;
        }

        // 第二步: 循环删除所有度为1的顶点及相关的边，并将另外与这些边相关的其它顶点的度减一
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (degrees[i] == 1) queue.offer(i);
        }
        while (!queue.isEmpty()) {
            int u = queue.poll();
            degrees[u]--;
            for (int v : adjacentList[u]) {
                degrees[v]--;
                if (degrees[v] == 1) queue.offer(v);
            }
        }

        //第三步: 找到最后一个度为2所在的边
        for (int i = n-1; i >= 0; i--) {
            if (degrees[edges[i][0]-1] > 1 && degrees[edges[i][1]-1] > 1) return edges[i];
        }

        return null;
    }

    /**
     * 2021-01-14 可被 5 整除的二进制前缀
     * https://leetcode-cn.com/problems/binary-prefix-divisible-by-5/
     * 给定由若干 0 和 1 组成的数组 A。
     * 我们定义 N_i：从 A[0] 到 A[i] 的第 i 个子数组被解释为一个二进制数（从最高有效位到最低有效位）
     * @param A 1 <= A.length <= 30000 && A[i] 为 0 或 1
     * @return 返回布尔值列表 answer，只有当 N_i 可以被 5 整除时，答案 answer[i] 为 true，否则为 false。
     */
    public List<Boolean> prefixesDivBy5(int[] A) {
        List<Boolean> ret = new LinkedList<>();

        int val = 0;
        for (int a : A) {
            val <<= 1;
            val += a;

            if ((val % 5) == 0) ret.add(true);
            else ret.add(false);

            val %= 10;      //只跟个位数有关,因此只保留个位数
        }

        return ret;
    }

    /**
     * 2021-01-15 移除最多的同行或同列石头
     * n 块石头放置在二维平面中的一些整数坐标点上。每个坐标点上最多只能有一块石头。
     * 如果一块石头的 同行或者同列 上有其他石头存在，那么就可以移除这块石头。
     * 给你一个长度为 n 的数组 stones ，其中 stones[i] = [xi, yi] 表示第 i 块石头的位置，不会有两块石头放在同一个坐标点上
     * https://leetcode-cn.com/problems/most-stones-removed-with-same-row-or-column/
     * @param stones 1 <= stones.length <= 1000 && 0 <= x[i], y[i] <= 10^4
     * @return 返回 可以移除的石子 的最大数量。
     */
    public int removeStones(int[][] stones) {
        int n = stones.length;
        //创建一个数组存储并查集
        int[] sets = new int[n];
        for (int i = 0; i < n; i++) sets[i] = i;

        for(int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                //如果两点属于同行或同列,合并两个集合为一个集合
                if (stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1]) {
                    sets[stoneSetFind(sets, i)] = stoneSetFind(sets,j);
                }
            }
        }

        //记录当前有多少个连通集合
        int sets_num = 0;
        for (int i = 0; i < n; i++) {
            if (sets[i] == i) sets_num++;
        }
        return n - sets_num;
    }
    private int stoneSetFind(int[] sets, int stone) {
        while (sets[stone] != stone) {
            stone = sets[stone];
        }
        return stone;
    }

    /**
     * 2021-01-16 打砖块
     * https://leetcode-cn.com/problems/bricks-falling-when-hit/
     * 有一个 m x n 的二元网格grid，其中 1 表示砖块，0 表示空白。
     * 砖块 稳定（不会掉落）的前提是： 一块砖直接连接到网格的顶部，或者至少有一块相邻（4 个方向之一）砖块 稳定 不会掉落时
     * 给你一个数组 hits ，这是需要依次消除砖块的位置。每当消除hits[i] = (rowi, coli) 位置上的砖块时，对应位置的砖块（若存在）会消失，
     * 然后其他的砖块可能因为这一消除操作而掉落。一旦砖块掉落，它会立即从网格中消失（即，它不会落在其他稳定的砖块上）。
     * @param grid m == grid.length && n == grid[i].length
     *             && 1 <= m, n <= 200 && grid[i][j] 为 0 或 1
     * @param hits 1 <= hits.length <= 4 * 10^4 && hits[i].length == 2
     *             0 <= x[i] <= m - 1 && 0 <= y[i] <= n - 1 && 所有 (x[i], y[i]) 互不相同
     * @return 返回一个数组 result ，其中 result[i] 表示第 i 次消除操作对应掉落的砖块数目。
     *         注意，消除可能指向是没有砖块的空白位置，如果发生这种情况，则没有砖块掉落。
     */
    public int[] hitBricks(int[][] grid, int[][] hits) {
        return null;
    }

    /**
     * 2021-01-17 缀点成线
     * https://leetcode-cn.com/problems/check-if-it-is-a-straight-line/
     * 在一个XY坐标系中有一些点，我们用数组coordinates来分别记录它们的坐标，其中coordinates[i] = [x, y]表示横坐标为 x、纵坐标为 y的点。
     * @param coordinates 2 <= coordinates.length <= 1000 && coordinates[i].length == 2
     *                    && -10^4 <= coordinates[i][0], coordinates[i][1] <= 10^4
     *                    && coordinates 中不含重复的点
     * @return 请你来判断，这些点是否在该坐标系中属于同一条直线上，是则返回 true，否则请返回 false。
     */
    public boolean checkStraightLine(int[][] coordinates) {
        if (coordinates.length == 2) return true;
        int z_x = coordinates[0][0];
        int z_y = coordinates[0][1];

        for (int i = 2; i < coordinates.length; i++) {
            int kii = (coordinates[i][1] - z_y) * (coordinates[i-1][0] - z_x);
            int ki = (coordinates[i-1][1] - z_y) * (coordinates[i][0] - z_x);
            if (ki != kii) return false;
        }

        return true;
    }

    /**
     * 2021-01-18 账户合并
     * https://leetcode-cn.com/problems/accounts-merge/
     * 给定一个列表 accounts，每个元素 accounts[i]是一个字符串列表，
     * 其中第一个元素 accounts[i][0]是名称 (name)，其余元素是 emails 表示该账户的邮箱地址。
     * 现在，我们想合并这些账户。如果两个账户都有一些共同的邮箱地址，则两个账户必定属于同一个人。
     * 请注意，即使两个账户具有相同的名称，它们也可能属于不同的人，因为人们可能具有相同的名称。一个人最初可以拥有任意数量的账户，但其所有账户都具有相同的名称。
     * @param accounts 1 <= accounts.length <= 1000 && 1 <= accounts[i].length <= 10 && 1 <= accounts[i][j].length <= 30
     * @return 合并账户后，按以下格式返回账户：每个账户的第一个元素是名称，其余元素是按顺序排列的邮箱地址。账户本身可以以任意顺序返回。
     */
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        if (accounts.size() == 1) return accounts;

        // 利用一个邮箱账户的映射存储并查集
        Map<String, String> sets = new HashMap<>();
        // 建立邮箱到用户的映射
        Map<String, String> emailToName = new HashMap<>();
        for (List<String> account : accounts) {
            String name = account.get(0);
            int size = account.size();
            for (int i = 1; i < size; i++) {
                String email = account.get(i);
                if (!sets.containsKey(email)) {
                    // 如果并查集中没有这个邮箱，则添加邮箱其根元素就是本身
                    sets.put(email,email);
                    // 添加该邮箱对应的账户名映射
                    emailToName.put(email,name);
                }
                if(i > 1) {
                    // 并查集的union操作，合并一个账户中的所有邮箱
                    sets.put(findAccount(sets,account.get(i)), findAccount(sets,account.get(1)));
                }
            }
        }

        // 暂时存储答案中的邮箱列表，每个键值对的键就是每个并查集集合的根元素
        Map<String,List<String>> merge = new HashMap<>();
        for (String email : sets.keySet()) {
            // 获取当前邮箱对应并查集的根元素
            String root = findAccount(sets,email);
            // 将当前邮箱放入根元素对应的列表中
            if(!merge.containsKey(root)) merge.put(root, new ArrayList<>());
            merge.get(root).add(email);
        }

        List<List<String>> res = new ArrayList<>();
        // 将答案从映射中放到列表总
        for(String root : merge.keySet()) {
            // 获取当前根元素对应的列表
            List<String> layer = merge.get(root);
            // 题目要求的排序
            Collections.sort(layer);
            // 添加姓名
            layer.add(0, emailToName.get(root));
            // 将当前列表加入答案
            res.add(layer);
        }

        return res;
    }
    private String findAccount(Map<String, String> sets, String email) {
        while (!email.equals(sets.get(email))) {
            email = sets.get(email);
        }
        return email;
    }

    /**
     * 2021-01-19 连接所有点的最小费用
     * https://leetcode-cn.com/problems/min-cost-to-connect-all-points/
     * 给你一个points 数组，表示 2D 平面上的一些点，其中 points[i] = [xi, yi] 。
     * 连接点[xi, yi] 和点[xj, yj]的费用为它们之间的 曼哈顿距离：|xi - xj| + |yi - yj|，其中|val|表示val的绝对值。
     * @param points 1 <= points.length <= 1000 && -10^6 <= xi, yi <= 10^6 && 所有点 (xi, yi) 两两不同
     * @return 请你返回将所有点连接的最小总费用。只有任意两点之间 有且仅有 一条简单路径时，才认为所有点都已连接。
     */
    public int minCostConnectPoints(int[][] points) {
        int n = points.length;
        if (n < 2) return 0;

        Queue<Edge> edgesQueue = new PriorityQueue<>(new Comparator<Edge>() {
            @Override
            public int compare(Edge o1, Edge o2) {
                return o1.dist - o2.dist;
            }
        });

        /*将所有点与点之间建立边关系*/
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int i_j_dist = manhattanDist(points[i][0], points[i][1], points[j][0], points[j][1]);
                edgesQueue.add(new Edge(i_j_dist, i, j));
            }
        }

        int minCostDist = 0;
        int count = 1;
        /*建立一个结合存储并查集*/
        int[] sets = new int[n];
        for (int i = 0; i < n; i++) sets[i] = i;
        while (!edgesQueue.isEmpty()) {
            Edge edge = edgesQueue.poll();
            //进行union操作
            int s = pointSetFind(sets, edge.origin);
            int t = pointSetFind(sets, edge.destine);
            if (s != t) {
                sets[s] = t;
                minCostDist += edge.dist;
                count++;
            }
            if (count == n) break;
        }

        return minCostDist;
    }
    private int pointSetFind(int[] sets, int point) {
        while (sets[point] != point) {
            sets[point] = sets[sets[point]];
            point = sets[point];
        }
        return point;
    }
    /**图边的数据结构*/
    private static class Edge {
        int origin; //起始点
        int destine; //终点
        int dist;
        Edge(int dist, int start, int end){
            origin = start;
            destine = end;
            this.dist = dist;
        }
    }
    /**曼哈顿距离*/
    private int manhattanDist (int xi, int yi, int xj, int yj) {
        return Math.abs(xi - xj) + Math.abs(yi - yj);
    }

    /**
     * Pow(x, n)
     * https://leetcode-cn.com/problems/powx-n/
     * @param x -100.0 < x < 100.0
     * @param n n 是 32 位有符号整数，其数值范围是 [−2^31, 2^31 − 1]
     * @return 计算 x 的 n 次幂函数。
     */
    public double myPow(double x, int n) {
        if (n == 0) return 1.0;
        if (n == 1) return x;

        if (n < 0) {
            return ((1 / x) * myPow(1 / x, -(n + 1)));
        }

        //当n为偶数: 2^2 = 4^1 当n为奇数: 2^3 = 2*4^1
        return ((n & 1) == 0) ? myPow(x * x, n >> 1) : x * myPow(x * x, n >> 1);
    }

    /**
     * 2021-01-20 三个数的最大乘积
     * https://leetcode-cn.com/problems/maximum-product-of-three-numbers/
     * 给定一个整型数组，在数组中找出由三个数组成的最大乘积，并输出这个乘积。
     * @param nums 给定的整型数组长度范围是[3,10^4]，数组中所有的元素范围是[-1000, 1000]。
     * @return 输入的数组中任意三个数的乘积不会超出32位有符号整数的范围。
     */
    public int maximumProduct(int[] nums) {
        /*Arrays.sort(nums);*/  //O(NLogN)
        //int n = nums.length;
        /*
        * 排序之后最大乘积就两种情况：
        * 1、如果全是正数就是最后三个数相乘
        * 2、如果有负数最大的乘机要么是最后三个数相乘，要么是两个最小的负数相乘再乘以最大的正数
        * */
        //return Math.max(nums[0]*nums[1]*nums[n-1] , nums[n-1]*nums[n-2]*nums[n-3]);

        // 最小的和第二小的
        int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        // 最大的、第二大的和第三大的
        int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, max3 = Integer.MIN_VALUE;

        for (int x : nums) {
            if (x < min1) {
                min2 = min1;
                min1 = x;
            } else if (x < min2) {
                min2 = x;
            }

            if (x > max1) {
                max3 = max2;
                max2 = max1;
                max1 = x;
            } else if (x > max2) {
                max3 = max2;
                max2 = x;
            } else if (x > max3) {
                max3 = x;
            }
        }

        return Math.max(min1 * min2 * max1, max1 * max2 * max3);
    }

    /**
     * 平衡括号字符串的最少插入次数
     * https://leetcode-cn.com/problems/minimum-insertions-to-balance-a-parentheses-string/
     * 给你一个括号字符串 s ，它只包含字符 '(' 和 ')' 。
     * 一个括号字符串被称为平衡的当它满足：
     *      任何左括号 '(' 必须对应两个连续的右括号 '))' 。
     *      左括号 '(' 必须在对应的连续两个右括号 '))' 之前。
     * 你可以在任意位置插入字符 '(' 和 ')' 使字符串平衡。
     * @param s 1 <= s.length <= 10^5 && s[i]只包含'('和')'
     * @return 请你返回让 s 平衡的最少插入次数。
     */
    public int minInsertions (String s) {
        int leftBracket = 0;    //记录多余的左括号数量,需要两个右括号与之匹配
        int insertions = 0;     //记录插入括号数量,包括左右两种括号

        for (int i = 0; i < s.length(); i++) {
            char bracket = s.charAt(i);
            //如果遇到的是左括号
            if (bracket == '(') {
                leftBracket++;
            }
            //如果遇到的是右括号
            else {
                // 先看看是否有库存的左括号与之匹配
                if (leftBracket == 0) insertions++;
                else leftBracket--;

                // 以下两种情况只有一个右括号，需要再加一个右括号
                if (i == s.length() - 1 || s.charAt(i + 1) != ')') insertions++;
                else i++;
            }
        }

        return insertions + leftBracket * 2;
    }

    /**
     * 负二进制转换
     * https://leetcode-cn.com/problems/convert-to-base-2/
     * @param N 0 <= N <= 10^9
     * @return 给出数字 N，返回由若干 "0" 和 "1"组成的字符串，该字符串为 N 的负二进制（base -2）表示。
     * 除非字符串就是 "0"，否则返回的字符串中不能含有前导零。
     */
    public String baseNeg2(int N) {
        //负二进制表示法: https://blog.csdn.net/u012140251/article/details/109409015

        if (0 == N) return "0";
        int BASE = -2;
        StringBuilder stb = new StringBuilder();

        while (N != 1) {
            int remainder = N % BASE;
            if (N > 0) {
                stb.append(remainder);
                N /= BASE;
            } else {
                if (remainder == -1) {
                    //如果商为负数并且余数为-1,需保证商为正并且余数为1
                    N = N/BASE + 1;
                    stb.append(1);
                } else {
                    stb.append(remainder);
                    N /= BASE;
                }
            }
        }

        // 最后需要处理商为1,因为1/(-2)=0,余1
        stb.append(1);
        stb.reverse();
        return stb.toString();
    }

    /**
     * 2021-01-21 找到最小生成树里的关键边和伪关键边
     * 给你一个 n个点的带权无向连通图，节点编号为 0到 n-1，同时还有一个数组 edges，
     * 其中 edges[i] = [fromi, toi, weighti]表示在fromi和toi节点之间有一条带权无向边。
     * 最小生成树(MST) 是给定图中边的一个子集，它连接了所有节点且没有环，而且这些边的权值和最小。
     *
     * https://leetcode-cn.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
     * @param n 2 <= n <= 100
     * @param edges 1 <= edges.length <= min(200, n * (n - 1) / 2)
     *              && edges[i].length == 3
     *              && 0 <= fromi < toi < n
     *              && 1 <= weighti <= 1000
     *              && 所有 (fromi, toi) 数对都是互不相同的。
     * @return 请你找到给定图中最小生成树的所有关键边和伪关键边。
     * 如果从图中删去某条边，会导致最小生成树的权值和增加，那么我们就说它是一条关键边。
     * 伪关键边则是可能会出现在某些最小生成树中但不会出现在所有最小生成树中的边。
     * 请注意，你可以分别以任意顺序返回关键边的下标和伪关键边的下标。
     */
    public List<List<Integer>> findCriticalAndPseudoCriticalEdges(int n, int[][] edges) {
        return null;
    }

    /**
     * 2021-01-22 数组形式的整数加法
     * https://leetcode-cn.com/problems/add-to-array-form-of-integer/
     * 对于非负整数X而言，X的数组形式是每位数字按从左到右的顺序形成的数组。例如，如果X = 1231，那么其数组形式为[1,2,3,1]。
     * 如果 A.length > 1，那么 A[0] != 0
     * @param A 1 <= A.length <= 10000 && 0 <= A[i] <= 9
     * @param K 0 <= K <= 10000
     * @return 给定非负整数 X 的数组形式A，返回整数X+K的数组形式。
     */
    public List<Integer> addToArrayForm(int[] A, int K) {
        List<Integer> list = new LinkedList<>();

        int a = A.length-1;
        int carry = 0;  //保留进位
        while (a >= 0 || K > 0) {
            //防止数组越界
            int rA = a >= 0 ? A[a] : 0;
            int rK = K%10;
            K /= 10;

            list.add(0,(rK+rA+carry)%10);

            if (rK+rA+carry >= 10)  carry = 1;
            else carry = 0;
            a--;
        }
        if (carry > 0) list.add(0,carry);

        return list;
    }

    /**
     * 2021-01-23 连通网络的操作次数
     * 用以太网线缆将 n 台计算机连接成一个网络，计算机的编号从 0 到 n-1。
     * 线缆用 connections 表示，其中 connections[i] = [a, b] 连接了计算机 a 和 b。
     * 网络中的任何一台计算机都可以通过网络直接或者间接访问同一个网络中其他任意一台计算机。
     * https://leetcode-cn.com/problems/number-of-operations-to-make-network-connected/
     * @param n 1 <= n <= 10^5
     * @param connections 1 <= connections.length <= min(n*(n-1)/2, 10^5)
     *                    && connections[i].length == 2
     *                    && 0 <= connections[i][0], connections[i][1] < n
     *                    && connections[i][0] != connections[i][1]
     *                    && 没有重复的连接。
     *                    && 两台计算机不会通过多条线缆连接。
     * @return 给你这个计算机网络的初始布线connections，你可以拔开任意两台直连计算机之间的线缆，并用它连接一对未直连的计算机。
     * 请你计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回-1。
     */
    public int makeConnected(int n, int[][] connections) {
        //n个点至少需要保证有n-1条边
        if (connections.length < n-1) return -1;

        //设置一个集合存储并查集
        int[] sets = new int[n];
        for (int i = 0; i < sets.length; i++) sets[i] = i;

        //将本来就相互连接着的电脑union成一个集合
        for (int[] connection : connections) {
            int begin = connection[0];
            int end = connection[1];
            int beginSet = connectedFind(sets, begin);
            int endSet = connectedFind(sets, end);
            if (beginSet != endSet) {
                sets[beginSet] = endSet;
            }
        }

        //看看还有多少台计算机没人爱
        int m = -1;
        for (int i = 0; i < sets.length; i++) {
            if (sets[i] == i) m++;
        }

        return m;
    }
    private int connectedFind(int[] sets, int i) {
        while (sets[i] != i) {
            sets[i] = sets[sets[i]];
            i = sets[i];
        }
        return i;
    }

    /**
     * 2021-01-24 最长连续递增序列
     * 连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，
     * 那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。
     * https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/
     * @param nums 0 <= nums.length <= 10^4 && -10^9 <= nums[i] <= 10^9
     * @return 给定一个未经排序的整数数组，找到最长且连续递增的子序列，并返回该序列的长度。
     */
    public int findLengthOfLCIS(int[] nums) {
        if (nums.length < 2) return nums.length;

        /*int[] dp = new int[nums.length];
        //base case
        dp[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = nums[i] > nums[i-1] ? dp[i-1]+1 : 1;
        }
        return Arrays.stream(dp).max().getAsInt();*/

        int lcis = 1;
        int idx = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i-1]) {
                lcis = Math.max(i+1-idx,lcis);
            }
            else {
                idx = i;
            }
        }

        return lcis;
    }

    /**
     * 2021-01-25 由斜杠划分区域
     * https://leetcode-cn.com/problems/regions-cut-by-slashes/
     * 在由 1 x 1 方格组成的 N x N 网格 grid 中，
     * 每个 1 x 1 方块由 '/' '\' 或 ' '构成。这些字符会将方块划分为一些共边的区域。
     * （请注意，反斜杠字符是转义的，因此 \ 用 "\\" 表示。）。
     * @param grid 1 <= grid.length == grid[0].length <= 30  && grid[i][j] 是 '/' '\' 或 ' '
     * @return 返回区域的数目。
     */
    public int regionsBySlashes(String[] grid) {
        //解决思路: 1 x 1的grid可以由 (grid.length || grid[0].length)=N 划分成N x N的小正方格
        //我们将N x N的小正方格中的每个N[i][j]再次划分成四个小区域顺时针编号N[i][j][0-3]
        //然后使用并查集连通区域集合,最后查看有多少个大的集合即可
        int N = grid.length;
        int[] sets = new int[N*N*4];
        for (int i = 0; i < sets.length; i++) sets[i] = i;

        for (int i = 0; i < N; i++) {
            String si = grid[i];
            for (int j = 0; j < N; j++) {
                //获取N[i][j][0]的集合编号
                int set_ij_0 = 4 * (i * N + j);
                char slash = si.charAt(j);
                switch (slash) {
                    case ' ' :
                        //如果是空格，需要合并N[i][j][0-3]四个小区域
                        slashUnion(sets,set_ij_0,set_ij_0+1);
                        slashUnion(sets,set_ij_0,set_ij_0+2);
                        slashUnion(sets,set_ij_0,set_ij_0+3);
                        break;
                    case '\\' :
                        //如果是'\'，需要分别合并 N[i][j][0]和N[i][j][1] N[i][j][2]和N[i][j][3]
                        slashUnion(sets,set_ij_0,set_ij_0+1);
                        slashUnion(sets,set_ij_0+2,set_ij_0+3);
                        break;
                    case '/' :
                        //如果是'/'，需要分别合并 N[i][j][0]和N[i][j][3] N[i][j][1]和N[i][j][2]
                        slashUnion(sets,set_ij_0,set_ij_0+3);
                        slashUnion(sets,set_ij_0+1,set_ij_0+2);
                        break;
                }
                //进行N[i][j]之间的合并无论是 '/' '\' 或 ' ' 都需要合并的地方
                if(i > 0) {
                    // 向上合并: 上一格(N[i][j-1][2]) --- 当前格(N[i][j][0])
                    slashUnion(sets, set_ij_0,set_ij_0 - 4 * N + 2);
                }
                if(j > 0) {
                    // 向左合并: 左一格(N[i-1][j][1]) --- 当前格(N[i][j][3])
                    slashUnion(sets,set_ij_0 + 3,set_ij_0 - 3);
                }
            }
        }

        int regions = 0;
        for (int i = 0; i < sets.length; i++) {
            if (sets[i] == i) regions++;
        }
        return regions;
    }
    private int slashFind(int[] sets, int i) {
        while (sets[i] != i) {
            sets[i] = sets[sets[i]];
            i = sets[i];
        }
        return i;
    }
    private void slashUnion(int[] sets, int i, int j) {
        int i1 = slashFind(sets, i);
        int j1 = slashFind(sets, j);
        if (i1 != j1) {
            sets[i1] = j1;
        }
    }

    /**
     * 2021-01-26 等价多米诺骨牌对的数量
     * https://leetcode-cn.com/problems/number-of-equivalent-domino-pairs/
     * 给你一个由一些多米诺骨牌组成的列表 dominoes。
     * 如果其中某一张多米诺骨牌可以通过旋转 0 度或 180 度得到另一张多米诺骨牌，我们就认为这两张牌是等价的。
     * 形式上，dominoes[i] = [a, b]和dominoes[j] = [c, d]等价的前提是a==c且b==d，或是a==d 且b==c。
     * 在0 <= i < j < dominoes.length的前提下，找出满足dominoes[i] 和dominoes[j]等价的骨牌对 (i, j) 的数量。
     * @param dominoes 1 <= dominoes.length <= 40000 && dominoes[i][j]取值[1,9]
     * @return 等价多米诺骨牌对的数量
     */
    public int numEquivDominoPairs(int[][] dominoes) {
        //由于dominoes[i][j]的取值是个位数,dominoes[i].length=2
        //两个个位数合并最大也才99,可以用数组替换哈希表
        int[] map = new int[100];

        int pair = 0;
        for (int[] domino : dominoes) {
            //将 两个个位数 凑成 一个十位数
            int val = domino[0] < domino[1] ? domino[0] * 10 + domino[1] : domino[1] * 10 + domino[0];
            pair += map[val];       //等同于 pair += map[i] * (map[i] - 1) / 2;  组合公式
            map[val]++;
        }

        return pair;
    }

    /**
     * 2021-03-04 LRU 缓存
     * https://leetcode-cn.com/problems/lru-cache-lcci/
     * 设计和构建一个“最近最少使用”缓存，该缓存会删除最近最少使用的项目。
     * 缓存应该从键映射到值(允许你插入和检索特定键对应的值)，并在初始化时指定最大容量。
     * 当缓存被填满时，它应该删除最近最少使用的项目。
     */
     private static class LRUCache {
         //解题思路: 双向链表+哈希表
        //双向链表: 新的放左边 <--> 旧的放右边
        final int capacity;
        Map<Integer,DualListNode> cache;
        DualListNode dummy_head;
        DualListNode dummy_tail;


        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.cache = new HashMap<>();
            dummy_head = new DualListNode(-1, -1);
            dummy_tail = new DualListNode(-1, -1);
            dummy_head.next = dummy_tail;
            dummy_tail.prev = dummy_head;
        }

        public int get(int key) {
            DualListNode node = cache.get(key);
            //如果不存在
            if (node == null) {
                return -1;
            }
            // 如果存在,先删除该节点原本位置,再移到最左边
            node.prev.next = node.next;
            node.next.prev = node.prev;
            moveToHead(node);
            return node.value;
        }

        // 把此节点标记为最新使用（放到最左边）
        private void moveToHead(DualListNode node) {
            node.next = dummy_head.next;
            node.next.prev = node;
            node.prev = dummy_head;
            dummy_head.next = node;
        }

        public void put(int key, int value) {
            if (this.get(key) == -1) {
                //如果此值不存在,加入并移到最左边
                DualListNode nNode = new DualListNode(key, value);
                cache.put(key,nNode);
                moveToHead(nNode);
                //如果容量超出上限,删除双向链表的最右边节点
                if (cache.size() > capacity) {
                    cache.remove(dummy_tail.prev.key);
                    dummy_tail.prev = dummy_tail.prev.prev;
                    dummy_tail.prev.next = dummy_tail;
                }
            }
            else {
                //如果此值已存在,更新为最新,无需左移因为this.get(key)方法移动过了
                cache.get(key).value = value;
            }
        }

        //双向链表数据结构
        private static class DualListNode {
            DualListNode prev;
            DualListNode next;
            int key;
            int value;
            public DualListNode(int key,int value) {
                this.key = value;
                this.value = value;
            }
        }
    }

    /**
     * 2022-02-23 编辑距离
     * 给你两个单词 word1 和 word2
     * 提示：
     *     0 <= word1.length, word2.length <= 500
     *     word1 和 word2 由小写英文字母组成
     * 你可以对word1进行如下三种操作:  插入一个字符 删除一个字符 替换一个字符
     * https://leetcode-cn.com/problems/edit-distance/
     * @return 请返回将 word1 转换成 word2 所使用的最少操作数。
     */
    public int minDistance(String word1, String word2) {
        //解决方法: 动态规划
        //假设dp[i][j]表示将word1前i个 -> word2前j个的最少操作数
        //定义如下三种操作:
        //1替换. dp[i-1][j-1]表示(将word1前i-1个 -> word2前j-1个)所需最少操作数
        //2插入. dp[i][j-1]表示(将word1前i个 -> word2前j-1个，然后尾部插入word2的第j个)所需最少操作数
        //3删除. dp[i-1][j]表示(将word1前i-1个 -> word2前j个，然后删除word1的第i个)所需最少操作数
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int j = 0; j < word2.length(); j++) {
            dp[0][j] = j;   //只能不停地插入操作
        }
        for (int i = 0; i < word1.length(); i++) {
            dp[i][0] = i;   //只能不停地删除操作
        }

        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                //如果word1[i] == word2[j], 此时最少操作数等同求dp[i-1][j-1]
                if (word1.charAt(i) == word2.charAt(j)) dp[i][j] = dp[i-1][j-1];
                //否则取 min(替换,插入,删除) + 1 作为最少操作数
                else {
                    dp[i][j] = Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1])) + 1;
                }
            }
        }
        return dp[word1.length()][word2.length()];
    }


}