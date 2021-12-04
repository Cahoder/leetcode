import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * 面试/笔试问题汇总
 */
public class InterviewQuestion {
    /*携程笔试  2021-3-4 */
    /*
    题目描述：
    给定一个表达式，求其计算结果。
    表达式的结构是这样的形式：(operator operand operand …)
    1、左右括号分别标志了表达式的开始和结束。
    2、operator是操作符，表示了计算规则，取值有三种:+-*，分别是加法、减法、乘法。
    3、operand是操作数，它既可以是一个整数，也可以是另一个表达式。操作数至少是两个。
    4、括号两边有0个或者多个空格。operator、operand之间有1个或者多个空格。
    计算规则如下：
    1、	如果operand是一个表达式，要先计算其值，再用该值参与运算。
    2、	如果operator是加法或者乘法，把所有operand相加或者相乘。
    3、	如果operator是减法，第一个operand是被减数，其他均为减数。
    下面几个例子演示了求值过程和结果：
    (+ 3 4) => 7
    (- 9 4 5) => 0
    (- (* 4 5) 4 5) => (-20 4 5) => 11
    (*(+2 3) (-100 (+ 20 10))) => (* 5 (-100 30)) => (* 5 (-100 30)) => (* 5 70)=> 350
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        StringBuilder exp = new StringBuilder(sc.nextLine().trim());
//
//        //配对括号
//        int left,right;
//        while (exp.lastIndexOf("(") != -1) {
//            left = exp.lastIndexOf("(");
//            right = exp.indexOf(")",left);
//            String str = exp.substring(left + 1, right).trim();
//
//            String op = str.substring(0,1);
//            String[] s = str.substring(1).trim().split(" ");
//            long count = 0;
//            if (op.equals("+")) {
//                for (int i = 0; i < s.length; i++) {
//                    count += Integer.parseInt(s[i]);
//                }
//            }
//            else if (op.equals("-")) {
//                count = Integer.parseInt(s[0]);
//                for (int i = 1; i < s.length; i++) {
//                    count -= Integer.parseInt(s[i]);
//                }
//            }
//            else {
//                count = 1;
//                for (int i = 0; i < s.length; i++) {
//                    count *= Integer.parseInt(s[i]);
//                }
//            }
//            exp.replace(left,right+1,String.valueOf(count));
//        }
//
//        System.out.println(exp);
//    }


    /*
    新春红包礼盒由一些额度不完全相同的小红包组成，用数组 packets 来表示每一份小红包的额度。
    你打算和 n 位朋友一同分享红包，所以你将红包拆分为 n + 1 份，每一份都由一些连续的小红包组成。
    作为发起人，你总是会拿取总额最小的那份，剩余的几份由朋友随机抽取。
    请找出一个最佳的拆分策略，使得你所分得的红包总额在可能的拆分策略中最大，返回这个最大总额。

    输入：
        [1,2,3,4,5,6,7,8,9]
        5
        解释：
        第一行数组 packets 数值表示红包金额，
        第二行数值表示朋友数量 n
    输出：6
    解释： 你可以把红包拆分成 [1,2,3], [4,5], [6], [7], [8], [9]，共6份，你拿总额最小的那份即[1,2,3]，其总额为1+2+3=6。

    提示
    1. 将数组分割成n+1组， 每组为一个连续子序列
    2. 求每组子序列和， 使得最小那组子序列之和在所有可能的分割方案中值最大
     */
    /*请完成下面这个函数，实现题目要求的功能
     ******************************开始写代码******************************/
    static int maxAmount(int[] packets, int n) {
        Arrays.sort(packets);
        int[] friends = new int[n];
        for (int i = 0; i < friends.length; i++) {
            friends[i] = packets[i+n-1];
        }
        int[] mines = new int[packets.length-n];
        for (int i = 0; i < mines.length; i++) {
            mines[i] = packets[i];
        }

        //不停地把我的红包里最大钞票分给除了我之外红包最少的那位朋友
        while (Arrays.stream(mines).sum() > friends[0]) {
            friends[0] += mines[mines.length-1];
            mines[mines.length-1] = 0;
            Arrays.sort(mines);
            Arrays.sort(friends);
        }

        return Arrays.stream(mines).sum();
    }

    /*汉得笔试 2021-3-6 */
    /*
     * 某城市环路设有新能源充电站点N个，其中第i个充电站点可供充电的电量为electronic[i]Kwh，
     * 现假设，你有一辆可以无限充电的电动汽车，从第i个充电桩开往第i+1个站点需要耗电cost[i]Kwh。
     * 该车从环路线路的某一充电站出发，开始时该电车空电量，若如果该电车可以绕环路行驶一周，则返回出发时充电站点编号，否则返回-1
     输入描述:
         输入数组均为非空数组，且长度相同。
         输入数组中的元素均为非负数。
         输入的两组数组中的间隔为空格
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//
//        String str = sc.nextLine();
//        String[] sss = str.split(" ");
//
//        String[] eles = sss[0].split(",");
//        String[] coss = sss[1].split(",");
//        if (eles.length != coss.length) {
//            System.out.println(-1);
//            return;
//        }
//
//        int[] electronic = new int[eles.length];
//        int[] cost = new int[coss.length];
//
//        for (int i = 0; i < eles.length; i++) {
//            electronic[i] = Integer.parseInt(eles[i]);
//        }
//        for (int i = 0; i < coss.length; i++) {
//            cost[i] = Integer.parseInt(coss[i]);
//        }
//
//        for (int i = 0; i < electronic.length; i++) {
//            //电量不足以消耗 跳过
//            if (electronic[i] < cost[i]) continue;
//
//            int volume = 0;
//            for (int j = i;;) {
//                //先充电
//                volume += electronic[j];
//                //再耗电
//                volume -= cost[j];
//
//                //半路没油了
//                if (volume < 0) break;
//                j++;
//                j %= electronic.length;
//                //绕完一圈了
//                if (j == i) break;
//            }
//
//            //跑完一圈还剩油
//            if (volume >= 0) {
//                System.out.println(i);
//                return;
//            }
//        }
//
//        System.out.println(-1);
//    }

    /*
    小李和同事玩贪吃蛇游戏。
    起始时，蛇位于m*n网格的左上角，现规定， 贪吃蛇每次只能向右或者向下挪动一格且不能回退，
    猎物在该网格的右下角。
    请问，此种情况下，有多少种走法可以吃到该猎物？
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String s = sc.nextLine();
//        String[] ss = s.split(",");
//        int m = Integer.parseInt(ss[0]);
//        int n = Integer.parseInt(ss[1]);
//
//        boolean[][] grid = new boolean[m][n];
//        dfs(grid,0,0);
//        System.out.println(count);
//    }
//
//    static int count = 0;
//    private static void dfs (boolean[][] grid, int x, int y) {
//        if (x >= grid[0].length || y >= grid.length) return;
//        if (x == grid[0].length-1 && y == grid.length-1) count++;
//
//        dfs(grid,x+1,y);
//
//        dfs(grid,x,y+1);
//    }

    /*笔试 奇安信 2021-3-6*/
    /**
     * 吃草少和产奶多会打架 吃草多和产奶少会打架
     * 吃草相同 或 产奶相同 不会打架
     * 求最大牛奶桶数
     * @param grass int整型一维数组 吃的草的捆数
     * @param milk int整型一维数组 产的牛奶桶数
     * @return int整型
     *
     * 作者：yuczzzzzz
     * 链接：https://www.nowcoder.com/discuss/607328?type=post&order=time&pos=&page=1&channel=-1&source_id=search_post_nctrack
     * 来源：牛客网
     * for循环嵌套，第一层for循环选第一只牛，第二层for循环取其他牛，并且用一个list记录取过的牛的索引，
     * 在第二层for循环里对将要取的牛和已经取过的牛的list比较，符合条件的就累加产奶量，把每次的结果存到产奶的list里，最后Collections.max(产奶list)输出结果
     */
    public static int MaxMilk (int[] grass, int[] milk) {
        if (grass.length < 2) return Arrays.stream(milk).sum();

        //先要根据奶量升序排序,同时改变草量相应数组位置
        for (int end = milk.length-1; end > 0 ; end--) {
            for (int begin = 1; begin <= end; begin++) {
                if (milk[begin] < milk[begin-1]) {
                    int tmp = milk[begin];
                    milk[begin] = milk[begin-1];
                    milk[begin-1] = tmp;

                    tmp = grass[begin];
                    grass[begin] = grass[begin-1];
                    grass[begin-1] = tmp;
                }
            }
        }

        int max = milk[0];
        for (int i = 1; i < milk.length; i++) {
            //奶同不打架
            if (milk[i] == milk[i-1]) {
                max += milk[i];
                continue;
            }

            //如果奶比你多但是用草比你少,会打架
            if (grass[i] < grass[i-1]) {
                max = Math.max(max,milk[i]);
            }
            else {
                max += milk[i];
            }
        }

        return max;
    }

    /*图的最短路径*/
//    public static void main(String[] args) {
//        //构建邻接矩阵
//        Scanner sc = new Scanner(System.in);
//        int vertex = sc.nextInt();
//        int begin = sc.nextInt();
//        int end = sc.nextInt();
//
//
//
//        while (true) {
//            int a = sc.nextInt();   //起点
//            int b = sc.nextInt();   //终点
//            int c = sc.nextInt();   //权重
//            if (a == 0 && b == 0 && c ==0) break;
//
//        }
//
//        System.out.println(count);
//    }
//    static int count = 0;

    /*笔试 富途 2021-3-6*/
    /**
     * 例: 2,1,5,3,4 -> 5,4,3,1,2
     * 栈排序 入栈顺序不变 进行从大到小出栈
     * 如果无法完全降序,请输出字典序最大的出栈序列
     * @param a int整型一维数组 描述入栈顺序
     * @return int整型一维数组
     */
    public static int[] solve (int[] a) {
        if (a == null) return null;
        if (a.length < 2) return a;

        int[] out = new int[a.length];
        int odx = 0;



        return out;
    }

    /**
     * 解码
     * 等同 https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/comments/
     * @param nums string字符串 数字串
     * @return int整型
     */
    public static int solve (String nums) {
        if (nums == null || nums.equals("")) return 0;
        return traceback(nums,0);
    }
    public static int traceback(String strs,int cur) {
        if (cur >= strs.length()) {
            return 1;
        }
        //如果能够解码2位
        if (cur+2 <= strs.length() && Integer.parseInt(strs.substring(cur,cur+2)) <= 26) {
            return traceback(strs,cur+1) + traceback(strs,cur+2);
        }
        //如果只能解码1位
        return traceback(strs,cur+1);
    }

    /*
        1. 三个同样的字母连在一起，一定是拼写错误，去掉一个的就好啦：比如 helllo -> hello
        2. 两对一样的字母（AABB型）连在一起，一定是拼写错误，去掉第二对的一个字母就好啦：比如 helloo -> hello
        3. 上面的规则优先“从左到右”匹配，即如果是AABBCC，虽然AABB和BBCC都是错误拼写，应该优先考虑修复AABB，结果为AABCC
     */
    public static String proofread(String strs) {
        StringBuilder stb = new StringBuilder(strs);
        //校对规则1
        for (int i = 0; i < stb.length()-2; i++) {
            if (stb.charAt(i) == stb.charAt(i+1) && stb.charAt(i+1) == stb.charAt(i+2)){
                stb.deleteCharAt(i);
                i--;
            }
        }
        //校对规则2
        for (int i = 0; i < stb.length()-3; i++) {
            if (stb.charAt(i) == stb.charAt(i+1) && stb.charAt(i+2) == stb.charAt(i+3)) {
                stb.deleteCharAt(i+2);
                i--;
            }
        }

        return stb.toString();
    }

    /**
     * x给定N（可选作为埋伏点的建筑物数）、D（相距最远的两名特工间的距离的最大值）以及可选建筑的坐标，计算在这次行动中有多少种埋伏选择。
     * 注意：
     * 1. N个建筑中选定3个埋伏地点
     * 1. 两个特工不能埋伏在同一地点
     * 2. 三个特工是等价的：即同样的位置组合(A, B, C) 只算一种埋伏方法，不能因“特工之间互换位置”而重复使用
     * 3. 结果可能溢出，请对 99997867 取模
     */
    public long combination(int N, int[] cors, int D) {
        long ambush = 0;
        for(int i = 0;i < N; i++) {
            int start = i;int end = N-1;
            while(start <= end) {
                int mid = start + (end-start)/2;
                //由于数据是升序,可以利用二分缩小范围
                if(cors[mid] - cors[i] > D) end = mid-1;
                else start = mid+1;
            }
            end = end < 0 ? -1 : end;
            if((end-i) >= 2) ambush = (ambush + (long)(end-i-1)*(end-i)/2)%99997867;
        }
        return ambush;
    }

/*笔试 保融科技 2021-03-07*/
    /**
     * 观察者
     */
    static class Observer {
        private final Set<Observed> container = new HashSet<>();
        //加入观察
        public void add (Observed observed) {
            container.add(observed);
        }
        //移除观察
        public Observed remove (Observed observed) {
            for (Observed next : container) {
                if (next.equals(observed)) {
                    container.remove(observed);
                    return next;
                }
            }
            return null;
        }
        //通知被观察者
        public void toNotify () {
            for (Observed next : container) {
                next.todo();
            }
        }
    }

    /**
     * 被观察者
     */
    static class Observed {
        public void todo() {
            /*TODO*/
        }
    }

    //生产者消费者模式
//    public static void main(String[] args) {
//        AtomicInteger cakes = new AtomicInteger();
//        final Object lock = new Object();
//        Runnable Productor = () -> {
//            while (true) synchronized (lock) {
//                if (cakes.get() >= 3) {
//                    try {
//                        System.out.println("有蛋糕");
//                        lock.wait();
//                    } catch (InterruptedException e) {
//                        e.printStackTrace();
//                    }
//                }
//                System.out.println("没蛋糕,制作");
//                cakes.getAndIncrement();
//                lock.notify();
//            }
//        };
//        Runnable Consumer = () -> {
//            while (true) synchronized (lock) {
//                if (cakes.get() < 3) {
//                    try {
//                        System.out.println("没蛋糕");
//                        lock.wait();
//                    } catch (InterruptedException e) {
//                        e.printStackTrace();
//                    }
//                }
//                System.out.println("有蛋糕,吃");
//                cakes.getAndDecrement();
//                lock.notify();
//            }
//        };
//        new Thread(Productor).start();
//        new Thread(Consumer).start();
//    }


/*笔试 顺丰科技 2021-03-10*/
    /*
    题目描述：
        现在有一行n个评委给选手打分，打出的分数分别为a1,a2,…an。
        我们知道，评委打分可能会出现极端情况，为了减少这样极端情况的影响，需要除开一个最高分和一个最低分，再对剩下的分数求平均值。
        现在有q次询问，每次给出L,R，请问如果选aL,aL+1,…aR这些评委分数作为最终打分依据，选手能得多少分？（即去掉aL,aL+1,…aR中最高分和最低分后的平均值）

    输入描述:
        第一行两个以空格隔开的正整数n,q，代表评委数以及询问数
        第二行n个以空格隔开的整数ai，依次代表第一个、第二个…第n个评委给出的分数
        接下来q行，每行两个以空格隔开的正整数L,R，含义如上。
        对于100%的数据，n,q≤40000，0≤ai≤104，1≤L≤R≤n
    输出描述:
        输出q行，每行依次代表一个询问的答案，如果选aL,aL+1,…aR这些评委分数作为最终打分依据，选手能得的分数
        （即去掉aL,aL+1,…aR中最高分和最低分后的平均值）。
        (向下取整，保留整数) 如果选出的分数不足三个，请输出”NoScore”(不含引号)。
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int q = sc.nextInt();
//        int[] scores = new int[n];
//
//        for (int i = 0; i < n; i++) {
//            scores[i] = sc.nextInt();
//        }
//
//        for (int i = 0; i < q; i++) {
//            int l = sc.nextInt();
//            int r = sc.nextInt();
//
//            if (r - l + 1 < 3) System.out.println("NoScore");
//            else {
//                int sum = 0;
//                int min = Integer.MAX_VALUE;
//                int max = Integer.MIN_VALUE;
//                for (int j = l; j <= r; j++) {
//                    min = Math.min(min,scores[j-l]);
//                    max = Math.max(max,scores[j-l]);
//                    sum += scores[j-l];
//                }
//                System.out.println((sum-min-max) / (r - l - 1));
//            }
//        }
//
//        sc.close();
//    }

    /*
    题目描述：
        港口新到了n个货物，工人们需要将它们通过货车运送到公司。
        货物会先后到达港口，第i个到达港口的货物是第i号，价值是a[i]。
        每辆货车可以将编号连续的货物一起运输，花费为该车货物价值的和的平方。
        货车有10种型号，均有足够多辆，第i种型号的货车可以容纳至多i个货物，由于不同型号货车所在位置不同，
        故每调用新型号的车（之前没有调用过这种型号），就得支付b[i]的成本。
        你是运输货车公司的老板，负责将全部货物运送到公司，你想知道最大利润，即花费减去运输成本的最大值是多少。

    输入描述：
        第一行一个数n。
        接下来n个数a[]，第i个数为a[i]。
        接下来10个数b[], 第i个数为b[i]。
    输出描述：
        一个数表示答案。
        如果最大利润为负，则输出0。

    样例输入：
        2
        5 5
        10 30 100 100 100 100 100 100 100 100
    样例输出：
        70

    提示：
        1≤n≤300
        0≤a[i]≤100
        0≤b[i]≤100,000,000
        只选1号车型，答案为40；
        只选2号车型，答案为70；
        选1和2号车型，答案为60（调用了两种类型，但是最终只用一辆二号车运载所有货物，100-10-30=60）.
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int[] package_values = new int[n];
//        for (int i = 0; i < n; i++) {
//            package_values[i] = sc.nextInt();
//        }
//        int[] car_costs = new int[10];
//        for (int i = 0; i < 10; i++) {
//            car_costs[i] = sc.nextInt();
//        }
//
//    }

/*笔试 跟谁学 2021-03-11*/
    /*
        不使用库函数，实现对输入的整数求平方根，结果四舍五入精确到小数点后3位。
    */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//
//        double l = 0,r = n;
//        while (l < r) {
//            double mid = (l+r)/2;
//            if (Math.pow(mid,2) - n > 1e-6) {
//                r = mid;
//            }
//            else if (Math.pow(mid,2) < n) {
//                l = mid;
//            }
//            else {
//                System.out.printf("%.3f\n",mid);
//                break;
//            }
//        }
//    }

    /*
        第一行读入一个整数代表矩阵的阶数N
        之后读入N行，每行包含N个整数，共计N * N个整数代表矩阵的元素

        输出N行，每行包含N个整数，共计N * N个整数代表转置后的N阶矩阵   (对角置换)
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//
//        int[][] nn = new int[n][n];
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < n; j++) {
//                nn[i][j] = sc.nextInt();
//            }
//        }
//        sc.close();
//
//        boolean[][] read = new boolean[n][n];
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < n; j++) {
//                if (i == j) continue;
//                if (read[i][j]) continue;
//                int tmp = nn[i][j];
//                nn[i][j] = nn[j][i];
//                nn[j][i] = tmp;
//                read[j][i] = true;
//            }
//        }
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < n; j++) {
//                System.out.print(nn[i][j]);
//                if (j < n-1) System.out.print(" ");
//            }
//            System.out.print("\n");
//        }
//    }

    /*
    在跟谁学学习的途途小朋友有一天向老师求助，原来途途的父母要出差几天，走之前给途途留下了一些棒棒糖。
    途途决定每天吃的棒棒糖数量不少于前一天吃的一半，但是他又不想在父母回来之前的某一天没有棒棒糖吃，
    他想让老师帮他计算一下，他第一天最多能吃多少个棒棒糖。

    输入两个数字N和M，分别代表出差天数，与棒棒糖数量
    输出第一天最多能吃糖的数量
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int N = sc.nextInt();   //出差天数
//        int M = sc.nextInt();   //糖果数量
//        sc.close();
//
//        if (N > M) System.out.println(0);
//        if (N == M) System.out.println(1);
//
//
//    }


/*模拟 金山wps*/
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int m = sc.nextInt();
//        int[] unionSets = new int[n*m];
//        boolean[] visited = new boolean[n*m];
//
//        for (int i = 0; i < n; i++) {
//            String next = sc.next();
//            for (int j = 0; j < m; j++) {
//                if (next.charAt(j) == '0') unionSets[i*m+j] = -1;
//                else unionSets[i*m+j] = i*m+j;
//            }
//        }
//        sc.close();
//
//        for (int i = 0; i < unionSets.length; i++) {
//            if (unionSets[i] == i) {
//                union(unionSets,i,i,m,visited);
//            }
//        }
//        //System.out.println(Arrays.toString(unionSets));
//
//        long count = Arrays.stream(unionSets).distinct().filter(value -> value != -1).count();
//        System.out.println(count);
//    }
//    private static void union(int[] unionSets, int i, int root,int m,boolean[] visited) {
//        if (i < 0 || i >= unionSets.length || visited[i] || unionSets[i] == -1) return;
//        visited[i] = true;
//        unionSets[i] = root;
//        //左
//        union(unionSets,i-1,root,m,visited);
//        //右
//        union(unionSets,i+1,root,m,visited);
//        //上
//        union(unionSets,i-m,root,m,visited);
//        //下
//        union(unionSets,i+m,root,m,visited);
//    }


/*笔试 阿里实习 2021-03-12*/
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();   //电话通讯中可能涉及到的人数
//        int m = sc.nextInt();   //通讯录中的电话存储关系数量
//
//        //建立邻接表
//        Set<Integer>[] users = new Set[n];
//        for (int i = 0; i < users.length; i++) {
//            users[i] = new HashSet<>();
//        }
//
//        for (int i = 0; i < m; i++) {
//            int user = sc.nextInt();
//            users[user-1].add(sc.nextInt()-1);
//        }
//
//        int q = sc.nextInt();
//
//        for (int i = 0; i < q; i++) {
//            int from = sc.nextInt()-1;
//            int to = sc.nextInt()-1;
////            System.out.println("from: "+ from);
////            System.out.println("to: "+ to);
//            steps = Integer.MAX_VALUE;
//            minSteps(users,from,to,new ArrayList<>(),new boolean[n]);
//            System.out.println(steps == Integer.MAX_VALUE ? -1 : steps);
//        }
//    }

    static int steps = Integer.MAX_VALUE;
    public static void minSteps (Set<Integer>[] users, int from, int to, List<Integer> step, boolean[] visited) {
//        System.out.println("123123");
//        System.out.println("from: "+ from);
//        System.out.println("to: "+ to);
        if (from == to) {
            steps = Math.min(steps,step.size());
            return;
        }
        Set<Integer> from_contacts = users[from];
//        System.out.println(from_contacts);
        for (Integer contact : from_contacts) {
//            System.out.println("contact: " + contact);
            if (visited[contact]) continue;
            visited[contact] = true;
            step.add(contact);
            minSteps(users, contact, to, step, visited);
            step.remove(step.size()-1);
            visited[contact] = false;
        }
    }


//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int T = sc.nextInt();   //T组测试数据
//
//        for (int i = 0; i < T; i++) {
//            int n = sc.nextInt();   //n*n 矩阵
//            int m = sc.nextInt();   //m个战车
//            for (int j = 0; j < m; j++) {
//                //每个战车的初始位置
//                int mx = sc.nextInt();
//                int my = sc.nextInt();
//
//                //计算最少需要移动多少步骤
//                //能够让战车都在主对角线上(mx==my)
//                //战车之间不可同时在同一横线或竖线上
//
//            }
//        }
//        sc.close();
//    }


/*笔试 美团 2021-03-13*/
    /*
    题目描述：
       小美有一点懒惰， 她懒得学太多东西和做太多事情。有一次她躺在床上做一项作业时，发现答案都写歪了，请帮她翻转到正确位置。
       形式化地，给出一个n×m的二维数组，第 i 行第 j 列的数记为a[i][j]。现在要将这个二维数组沿着aii（1≤i≤min（n，m））翻转180°。

    输入描述：
        输入n+1行，第一行两个数n和m，表示二维数组的行数和列数。
        接下来n行，每行m个数，第 i 行第 j 个数表示二维数组中的数a[i][j]
        1≤n,m≤100，0≤a[i][j]≤1000000，均为整数
    输出描述：
        输出m行，表示翻转后的二维数组。
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int m = sc.nextInt();
//        int[][] result = new int[m][n];
//
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < m; j++) {
//                result[j][i] = sc.nextInt();
//            }
//        }
//
//        for (int[] ints : result) {
//            for (int j = 0; j < ints.length; j++) {
//                System.out.print(ints[j]);
//                if (j < ints.length - 1) System.out.print(" ");
//            }
//            System.out.println("\n");
//        }
//    }

    /*
    题目描述：
    小美过冬之前将很多数藏进一个仅包含小写英文字母的字符串。在冬天她想将所有数找回来，请帮帮她。
    给定一个字符串s，仅包含小写英文字母和数字，找出其中所有数。
    一个数是指其中一段无法延伸的连续数字串。
    如a1254b中仅包含1254这一个数，125则不是，因为125还可以向右延伸成1254。
    如果该字符串包含前导零，则抹掉前导零。
    例如a0125b1c0d00中包含四个数0125，1，0，00，按照规则抹掉前导零后，最终这个字符串包含的四个数为125，1，0，0。
    即，0125->125，00->0，其中1和0无前导零，无需变更。

    输入描述
        输入一行，一个仅包含小写英文字母和数字的字符串s。
        1≤s的长度≤1010
    输出描述
        输出若干行，表示找到的所有数。
        按从小到大的顺序输出。


    样例输入
        0he15l154lo87wor7l87d
    样例输出
        0
        7
        15
        87
        87
        154
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        char[] chars = sc.nextLine().trim().toCharArray();
//
//        Queue<Character> tmp = new LinkedList<>();
//        PriorityQueue<Integer> queue = new PriorityQueue<>(Integer::compareTo);
//        for (char c : chars) {
//            //System.out.println(c);
//            if (c >= '0' && c <= '9') {
//                tmp.offer(c);
//                //System.out.println(tmp);
//            } else {
//                //清除前导零
//                while (tmp.size() > 1 && tmp.peek() == '0') tmp.poll();
//                int number = -1;
//
//                while (!tmp.isEmpty()) {
//                    if (number == -1) number++;
//                    number = number*10 + Character.getNumericValue(tmp.poll());
//                }
//                //System.out.println("number: " + number);
//                if (number != -1)
//                    queue.offer(number);
//            }
//        }
//        //如果还没清完
//        while (tmp.size() > 1 && tmp.peek() == '0') tmp.poll();
//        int number = -1;
//
//        while (!tmp.isEmpty()) {
//            if (number == -1) number++;
//            number = number*10 + Character.getNumericValue(tmp.poll());
//        }
//        if (number != -1)
//            queue.offer(number);
//
//        while (!queue.isEmpty()) {
//            System.out.println(queue.poll());
//        }
//    }

    /*
    题目描述：
    小美正在统计她公司的数据。她想要知道一定时间段内的某种特征，因此，她将n条数据按时间排好序依次给出，想要知道以某条数据开始的一个连续数据段内数据的众数情况。
    形式化地，给出n个数a1,.....,an分别表示时刻1,2,....,n的数据值。
    小美想要知道连续K条数据的情况，即ai,...,ai+K-1。
    请你求出当i=1,2,...,n-K+1时，ai,...,ai+K-1这些数据中的"众数"。
    如果"众数"有多个，输出最小的那一个。
    注意，此处“众数”的定义与通常定义有些许区别。
    我们这样定义 “众数”：如果出现次数最多的数只有一个数，那么“众数”就是它；否则，众数为出现次数最多的数中，数值最小的那一个。
    例如，序列[1 2 3 3]中， 3出现了两次，其他数仅出现了一次，所以“众数”为3。
    序列[1 2 3]中，三个数出现次数都为1次，都是出现次数最多的数（不存在出现次数大于1的数），所以“众数”是其中的数值最小的1。
    序列[5 2 5 2 3 3 4]中，5、2、3均出现了2次，都是出现次数最多的数（不存在出现次数大于2的数），所以“众数”是其中的数值最小的2。

    输入描述
        第一行两个数n和K分别表示数据总量以及她想要了解的连续数据长度。
        第二行n个数a1,...,an，表示各个数据值。
        1≤K≤n≤105， 1≤ai≤109

    输出描述
        输出n-K+1行，每行一个数，依次表示从i=1到i=n-K+1时，ai,...,ai+K-1中的众数。如果众数有多个，输出最小的那一个。
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int k = sc.nextInt();
//        int[] ns = new int[n];
//        for (int i = 0; i < n; i++) {
//            ns[i] = sc.nextInt();
//        }
//
//        Map<Integer,Node> map = new HashMap<>();
//        for (int i = 0; i < n; i++) {
//            //System.out.println(ns[i]);
//            if (map.get(ns[i]) == null) map.put(ns[i],new Node(ns[i],1));
//            else map.get(ns[i]).count++;
//            if (i+1-k >= 0) {
//                //System.out.println(map);
//                int max = Integer.MIN_VALUE;
//                for (Integer ii : map.keySet()) {
//                    max = Math.max(map.get(ii).count,max);
//                }
//                System.out.println(max);
//                map.get(ns[i+1-k]).count--;
//            }
//        }
//
//    }

    static class Node {
        int ni;
        int count;

        public Node(int ni, int count) {
            this.ni = ni;
            this.count = count;
        }
    }

    /*
    题目描述：
           小团和蚂蚁们成为了好朋友。蚂蚁们现在打算爬上一颗树做游戏，这一个游戏的开始需要蚂蚁们组队之后商量战术。
           为了更好地交流，同一个队伍的蚂蚁们会站在树上的同一个节点。
           不同队伍的蚂蚁们会间隔至少一个空的节点，这样的话它们同组交流战术时不会被其他组听到。
           而树上每一个节点可以容纳的蚂蚁数量是有限的。
           为了更好玩以及游戏性，蚂蚁们希望在参加蚂蚁数最多的前提下，使得蚂蚁数最少的队伍的蚂蚁数尽可能多。
           蚂蚁们不能很好地解决这个问题，希望小团帮忙解决。
           给定一棵树，每个节点有一个权值。
           选择其中某些节点，满足被选中的节点两两不相邻。
           求在所有的选择方案中，最大化被选择节点权值之和的情况下，被选择节点权值最小值尽可能大。
           树是一种无向连通图，任意节点两两可达且无环。
    输入描述：
        第一行两个数n和m，分别表示树上节点个数和树的边数。
        第二行n个数 a1 ,..., an ,ai 表示第 i 个节点上可以容纳的最大蚂蚁数。
        接下来m行，每行两个数u,v，表示u和v节点直接相连。
        1≤n,m≤105,   0≤ai≤109，1≤u,v≤n
        保证无重边无自环，数据均为整数
    输出描述
        输出一行，包含两个数，分别表示题面描述中的最优方案中的最大权值之和以及在这个前提下最大化了的最小值。

    输入样例
        5 4
        3 4 1 4 9
        1 2
        1 3
        2 4
        3 5
    输出样例
        16 3
     */







/*爱客科技*/
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int[] ns = new int[n];
//        for (int i = 0; i < n; i++) {
//            ns[i] = sc.nextInt();
//        }
//        for (int i = 0; i < n; i++) {
//            int j = i-1;
//            while (true) {
//                if (j < 0) {
//                    System.out.print(-1);
//                    break;
//                } else if (ns[j] < ns[i]) {
//                    System.out.print(j);
//                    break;
//                } else j--;
//            }
//            System.out.print(" ");
//            int m = i+1;
//            while (true) {
//                if (m >= n) {
//                    System.out.print(-1);
//                    break;
//                } else if (ns[m] < ns[i]) {
//                    System.out.print(m);
//                    break;
//                } else m++;
//            }
//            System.out.print("\n");
//        }
//    }
//    static int count = 0;
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        List<Integer> list = new ArrayList<>(n);
//        dfs(list,n);
//        System.out.println(count);
//    }
//    public static void dfs(List<Integer> list, int n) {
//        System.out.println(list);
//        if (list.size() == n) {
//            count++;
//            return;
//        }
//        for (int i = 0; i < 2; i++) {
//            if (i == 0) {
//                if (list.size() == 0 || list.get(list.size()-1) == 0) continue;
//            }
//            list.add(i);
//            dfs(list, n);
//            list.remove(list.size()-1);
//        }
//    }

/*奇安信*/
    /*
        当前会议的日程表 看看当天最多能参加多少个会议?
        要求参加会议的时间段不能够重合
     */
    public int AttendMeetings (int[][] times) {
        Arrays.sort(times, (o1, o2) -> {
            //按照会议持续时长短的优先 其次再按照会议开始时间升序排
            if ((o1[1] - o1[0]) == (o2[1] - o2[0])) return o1[0] - o2[0];
            return (o1[1] - o1[0]) - (o2[1] - o2[0]);
        });

        for (int[] time : times) {
            System.out.println(Arrays.toString(time));
        }

        int count = 0;
        boolean[] visited = new boolean[times.length];
        for (int i = 0; i < times.length; i++) {
            if (visited[i]) continue;
            //如果两个会议之间有交集 去参加耗时最短的那个
            for (int j = i+1; j < times.length; j++) {
                if (isDiffTime(times[i],times[j])) break;
                else visited[j] = true;
            }

            count++;
        }
        System.out.println(Arrays.toString(visited));
        System.out.println(count);
        return count;
    }

    public boolean isDiffTime(int[] timeA, int[] timeB) {
        return timeB[0] >= timeA[1];
    }


//
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//
//        int x = sc.nextInt();   //资金
//        int m = sc.nextInt();   //商品种类
//
//        int[] limit = new int[m]; //每种商品购买限制数量
//        for (int i = 0; i < limit.length; i++) {
//            limit[i] = sc.nextInt();
//        }
//        int[] nowPrice = new int[m];    //每种商品当前价格
//        for (int i = 0; i < nowPrice.length; i++) {
//            nowPrice[i] = sc.nextInt();
//        }
//        int[] afterTenPrice = new int[m];   //每种商品十天后价格
//        for (int i = 0; i < afterTenPrice.length; i++) {
//            afterTenPrice[i] = sc.nextInt();
//        }
//
//        //首先需要凑出购买方案
//        //再从所有方案中找出10天后获益最大的
//        toBuy(x,nowPrice,afterTenPrice,limit,new int[m]);
//        System.out.println(maxProfit);
//    }
//
//    static int maxProfit = Integer.MIN_VALUE;
//    public static void toBuy(int x, int[] nowPrice,int[] afterTenPrice,int[] limit,int[] buys) {
//        System.out.println(Arrays.toString(buys));
//        if (x == 0) {
//            //System.out.println(Arrays.toString(buys));
//            int profit = 0;
//            for (int i = 0; i < buys.length; i++) {
//                if (buys[i] > 0) profit += buys[i] * afterTenPrice[i];
//            }
//            maxProfit = Math.max(maxProfit,profit);
//            return;
//        }
//        for (int i = 0; i < nowPrice.length; i++) {
//            if (x - nowPrice[i] >= 0 && buys[i] < limit[i]) {
//                buys[i]++;
//                toBuy(x-nowPrice[i],nowPrice,afterTenPrice,limit,buys);
//            }
//        }
//    }


/*数字广东*/
    //1. 实现块排
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String rawStr = sc.nextLine();
//        sc.close();
//        String[] rawArr = rawStr.split(",");
//        arr = new int[rawArr.length];
//        for (int i = 0; i < arr.length; i++) {
//            arr[i] = Integer.parseInt(rawArr[i]);
//        }
//
//        quickSort(0,arr.length);
//
//        for (int i = 0; i < arr.length; i++) {
//            System.out.print(arr[i]);
//            if (i != arr.length-1) System.out.print(",");
//        }
//    }
    static int[] arr;

    /**
     * 块排 [start,end)
     */
    private static void quickSort(int start, int end) {
        if (end - start < 2) return;

        int mid = pivot(start,end);
        quickSort(start,mid);
        quickSort(mid+1,end);
    }
    //找中间点
    private static int pivot(int start, int end) {
        int pivot = arr[start];
        end--;

        while (start < end) {
            //先从右往左找比它小的
            while (start < end) {
                if (arr[end] > pivot) end--;
                else {
                    arr[start++] = arr[end];
                    break;
                }
            }
            //再从左往右找比它大的
            while (start < end) {
                if (arr[start] < pivot) start++;
                else {
                    arr[end--] = arr[start];
                    break;
                }
            }
        }

        arr[start] = pivot;
        return start;
    }

    //2. 字符串压缩 利用字符重复出现次数进行压缩 若压缩后字符串长度没有变短则返回原字符串
    // 假设基于的原字符串中只包含大小写字母(a-z)
    public String compressString (String S) {
        if (S == null || S.length() <= 2) return S;

        StringBuilder stb = new StringBuilder();
        char[] chars = S.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            int count = 1;

            for (int j = i+1; j < chars.length; j++) {
                if (chars[j] != chars[i]) {
                    i = j-1;
                    break;
                } else count++;
                if (j == chars.length-1) {
                    i = j;
                    break;
                }
            }

            stb.append(chars[i]).append(count);
        }

        if (stb.length() >= S.length()) return S;
        return stb.toString();
    }

/*360*/
    /*
    题目描述：
    小马最近找到了一款打气球的游戏。
    每一回合都会有n个气球，每个气球都有对应的分值，第i个气球的分值为ai。
    这一回合内，会给小马两发子弹，但是由于小马的枪法不准，一发子弹最多只能打破一个气球，甚至小马可能一个气球都打不中。
    现给出小马的得分规则：
    1. 若小马一只气球都没打中，记小马得0分。
    2. 若小马打中了第i只气球，记小马得ai分。
    3. 若小马打中了第i只气球和第j只气球（i＜j），记小马得ai|aj分。
    （其中 | 代表按位或，按位或的规则如下：
    参加运算的两个数，按二进制位进行或运算，只要两个数中的一个为1，结果就为1。
    即 0|0=0,1|0=1，1|1=1。
    例：2|4即00000010|00000100=00000110，所以2|4=6 ）
    现在请你计算所有情况下小马的得分之和。

    输入描述
    第一行，一个整数n，表示此回合的气球数量。
    第二行，用空格分开的n个整数，第i个整数为ai，表示每个气球对应的分值。
    1≤n≤50000,1≤ai≤100000

    输出描述
    一行一个整数，代表所有情况下小马的得分之和。
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int[] ns = new int[n];
//        for (int i = 0; i < n; i++) {
//            ns[i] = sc.nextInt();
//        }
//        sc.close();
//
//        int result = Arrays.stream(ns).sum();
//        result += combination(ns,0);
//
//        System.out.println(result);
//    }
    public static int combination(int[] arr,int idx) {
        int or = 0;
        if (idx >= arr.length) return or;
        for (int i = idx+1; i < arr.length; i++) {
            or += arr[idx] | arr[i];
        }
        return or + combination(arr,idx+1);
    }

    /*
    题目描述：
    X星人发现了一个藏宝图，在藏宝图中标注了N个宝库的位置。这N个宝库连成了一条直线，每个宝库都有若干枚金币。
    X星人决定乘坐热气球去收集金币，热气球每次最多只能飞行M千米（假设热气球在飞行过程中并不会发生故障）此外，由于设计上的缺陷，热气球最多只能启动K次。
    X星人带着热气球来到了第1个宝库（达到第1个宝库时热气球尚未启动），收集完第1个宝库的金币后将启动热气球前往下一个宝库。
    如果他决定收集某一个宝库的金币，必须停下热气球，收集完之后再重新启动热气球。
    当然，X星人每到一个宝库是一定会拿走这个宝库所有金币的。
    已知每一个宝库距离第1个宝库的距离（单位：千米）和宝库的金币数量。
    请问X星人最多可以收集到多少枚金币？

    输入描述
    单组输入。
    第1行输入三个正整数N、M和K，分别表示宝库的数量、热气球每次最多能够飞行的距离（千米）和热气球最多可以启动的次数，三个正整数均不超过100，相邻两个正整数之间用空格隔开。
    接下来N行每行包含两个整数，分别表示第1个宝库到某一个宝库的距离（千米）和这个宝库的金币枚数。（因为初始位置为第1个宝库，因此第1个宝库所对应行的第1个值为0。）
    输入保证所有的宝库按照到第1个宝库的距离从近到远排列，初始位置为第1个宝库。

    输出描述
    输出一个正整数，表示最多可以收集的金币数。

    样例输入
    5 10 2
    0 5
    8 6
    10 8
    18 12
    22 15
    样例输出
    25
    提示
    X星人启动热气球两次，分别收集第1个、第3个和第4个宝库的金币，一共可以得到的金币总数为5+8+12=25枚。

     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int N = sc.nextInt();   //宝库的数量
//        int M = sc.nextInt();   //热气球单次最远飞行距离
//        int K = sc.nextInt();   //热气球最多可启动次数
//
//        int[][] n_to_n = new int[N][2];
//        for (int i = 0; i < N; i++) {
//            n_to_n[i][0] = sc.nextInt();    //表示第1个宝库到第i个宝库的距离
//            n_to_n[i][1] = sc.nextInt();    //第i个宝库的金币枚数
//        }
//        sc.close();
//
//        //dp[n][k][0]表示第n个宝藏启动了k次未降落时收集的金币数
//        //dp[n][k][1]表示第n个宝藏启动了k次要降落时收集的金币数
//        int[][][] dp = new int[N][K+1][2];
//        //默认收集出发点第1个宝藏的金币数
//        for (int j = 0; j < K+1; j++) {
//            dp[0][j][0] = n_to_n[0][1];
//            dp[0][j][1] = n_to_n[0][1];
//        }
//
//        for (int i = 1; i < N; i++) {
//            //如果飞不到那么提前结束了
//            if (n_to_n[i][0] - n_to_n[i-1][0] > M) {
//                int profit = 0;
//                for (int j = 0; j < K+1; j++) {
//                    profit = Math.max(profit,dp[i-1][j][1]);
//                }
//                System.out.println(profit);
//                return;
//            }
//            //首先不降落收集状态转移方程
//            dp[i][0][0] = dp[i-1][0][0];
//            dp[i][0][1] = Math.max(dp[i-1][0][1],n_to_n[i][1]);
//            //进行k次降落的状态转移方程
//            for (int j = 1; j < K+1; j++) {
//                dp[i][j][0] = Math.max(dp[i-1][j][0],dp[i-1][j-1][1]+n_to_n[i][1]);
//                dp[i][j][1] = Math.max(dp[i-1][j][1],dp[i-1][j][0]);
//            }
//        }
//
//        int profit = 0;
//        for (int j = 0; j < K+1; j++) {
//            profit = Math.max(profit,dp[N-1][j][1]);
//        }
//
//        System.out.println(profit);
//    }

/*声网 2021-03-31*/
    /*
    1. 给定一个整数 0 < n < 10000 且 n 不能被 2或5 整除
       那么肯定存在一个整数 m, m = c * n 且 c >= 1,
       且 m 的十进制表示中每一位都由1组成, 求解其中值最小的m的位数

       例如 输入n为3,则存在满足条件的m为111位数为3
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        sc.close();
//
//        int m = 1;
//        int mc= 1;
//
//        for (long c = 1; ; c++) {
//            if (c*n == m) break;
//            if (c*n > m) {
//                //System.out.println("m: " + m);
//                c = 1;
//                m = m*10 +1;
//                mc++;
//            }
//        }
//
//        //System.out.println(m);
//        System.out.println(mc);
//    }

    /*
    2. 在字符串中查找第一个只出现一次的字符
    例如 输入 abaccdeff  输出 b
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String s = sc.nextLine();
//        char[] chars = s.toCharArray();
//        sc.close();
//
//        LinkedHashMap<Character,Integer> map = new LinkedHashMap<>(s.length());
//        for (char c : chars) {
//            map.put(c,map.getOrDefault(c,0)+1);
//        }
//
//        for (Character a : map.keySet()) {
//            if (map.get(a) == 1) {
//                System.out.println(a);
//                break;
//            }
//        }
//    }

/*瑛太莱 2021-03-31*/
    /*
        1. 斜线打印二维数组
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int m = sc.nextInt();
//        int n = sc.nextInt();
//        int[][] arr = new int[m][n];
//
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < n; j++) {
//                arr[i][j] = sc.nextInt();
//            }
//        }
//        sc.close();
//
//        StringBuilder stb = new StringBuilder();
//
//        // 对角线及以上部分
//        // 第0列到n-1列
//        for (int k = 0; k < n; k++) {
//            for (int i = 0, j = k; i < m && j >= 0; i++, j--) {
//                stb.append(arr[i][j]).append(",");
//            }
//        }
//
//        // 对角线以下部分
//        // 第1行到m-1行
//        for (int k = 1; k < m; k++) {
//            for (int i = k, j = n - 1; i < m && j >= 0; i++, j--) {
//                stb.append(arr[i][j]).append(",");
//            }
//        }
//
//        if (stb.length() > 1) stb.deleteCharAt(stb.length()-1);
//
//        System.out.println(stb.toString());
//    }

    /*
        母羊生羊问题
        母羊会在第2/4年生 第5年死
        一开始只有1头母羊, 15年后有多少头母羊
     */
//    static int sheep = 1;
//    public static void main(String[] args) {
//        helper(15);
//        System.out.println(sheep);
//    }
//    public static void helper(int year) {
//        for (int i = 1; i <= year; i++) {
//            if (i == 2 || i == 4) {
//                sheep++;
//                helper(year-i);
//            }
//            if (i == 5) {
//                sheep--;
//                break;
//            }
//        }
//    }

    /*
       3. 计算器
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String s = sc.nextLine().replaceAll(" ","");
//        StringBuilder stb = new StringBuilder(s);
//        sc.close();
//
//        //先乘除
//        while (stb.indexOf("*") != -1 || stb.indexOf("/") != -1) {
//            int mul = stb.indexOf("*");
//            int div = stb.indexOf("/");
//            int i = mul != -1 ? mul : div;
//            int j = i-1;
//            while (j >= 0) {
//                char c = stb.charAt(j);
//                if (c >= '0' && c <= '9' && j > 0) j--;
//                else break;
//            }
//            int n = Integer.parseInt(stb.substring(j,i));
//
//            int k = i+1;
//            while (k < stb.length()) {
//                char c = stb.charAt(k);
//                if (c >= '0' && c <= '9') k++;
//                else break;
//            }
//            int m = Integer.parseInt(stb.substring(i+1,k));
//
//            stb.replace(j,k,String.valueOf(mul != -1 ? n*m : n/m));
//        }
//
//        //后加减
//        while (stb.indexOf("+") != -1 || stb.indexOf("-") != -1) {
//            int add = stb.indexOf("+");
//            int sub = stb.indexOf("-");
//            int i = add != -1 ? add : sub;
//            int j = i-1;
//            while (j >= 0) {
//                char c = stb.charAt(j);
//                if (c >= '0' && c <= '9' && j > 0) j--;
//                else break;
//            }
//            int n = Integer.parseInt(stb.substring(j,i));
//
//            int k = i+1;
//            while (k < stb.length()) {
//                char c = stb.charAt(k);
//                if (c >= '0' && c <= '9') k++;
//                else break;
//            }
//            int m = Integer.parseInt(stb.substring(i+1,k));
//
//            stb.replace(j,k,String.valueOf(add != -1 ? n+m : n-m));
//        }
//
//        System.out.println(stb.toString());
//    }

    /*
       4. 糖的递增排列
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        sc.close();
//
//        int sum = 0;
//        for (int i = 0; ; i++) {
//            sum += i;
//            if (sum >= n) {
//                System.out.println(i);
//                break;
//            }
//        }
//    }

/*度小满金融 2021-04-01*/
    /*
    1. 北京天坛
        题目描述：
        北京天坛的圜丘坛为古代祭天的场所，分上、中、下三层，上层中心有一块圆形石板（称为天心石），环绕天心石砌m块扇面形石板构成第一环，向外每环依次增加m块。
        下一层的第一环比上一层的最后一环多m块，向外每环依次增加m块。
        已知每层环数相同。
        现给出每层的环数n和每一环比上一环增加的块数为m，求总共有多少块扇面形石板？

        输入描述
        单行输入。
        两个正整数n和m，表示每层的环数和每一环比上一环增加的块数（n<1e5,m<1e5），两个正整数之间用空格隔开。

        输出描述
        输出扇面形石板的总数。

        样例输入
        2 9
        样例输出
        189
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int m = sc.nextInt();
//        sc.close();
//
//        long result = 0;
//        long c = m;
//        for (int i = 0; i < n*3; i++) {
//            result += c;
//            c += m;
//        }
//
//        System.out.println(result);
//    }

    /*
    2. 阶梯
        题目描述：
        你有n个箱子，它们的高度分别为ai，你想要用它们做出一个尽可能长的阶梯。
        但是你对最长的阶梯长度不感兴趣，你只对其方案数感兴趣。
        形式化地，给出长度为n的序列{ai}，从中挑选出子序列{bi}，满足对所有合法下标i,有b[i]＜b[i+1]成立（即单调递增）(如果子序列{bi}长度为1，亦视为满足此条件)。
        在这些子序列中，长度为最大长度的子序列有多少个？
        （子序列：某个序列的子序列是从最初序列通过去除某些元素但不破坏余下元素的相对位置（在前或在后）而形成的新序列。
        例如{2,3,5}是{1,2,3,4,5}的子序列，而{2,1}和{1,6}不是。
        我们认为两个子序列相同，当且仅当所有数都是从相同位置选出来的。
        而对于序列{1,2,2,6}，选择第2个和第4个形成子序列{2,6}，选择第3个和第4个形成子序列{2,6}，虽然形式相同但仍视为不同的序列，
        因为前者的2是原序列中第2个，后者的2是原序列中的第3个，并非相同位置选出）

        输入描述
        第一行一个正整数 n 表示序列长度。
        第二行 n 个由空格隔开的正整数 ai ，依次表示a1到an。
        对于100%的数据，1≤n≤3000，ai≤1000000000

        输出描述
        一行一个数，表示答案，对1000000007取模

        样例输入
        5
        1 3 6 4 7
        样例输出
        2
        提示
        显然，我们可以选取出长度为4的满足要求的子序列，可以证明，没有比4更长的满足要求的序列了。
        而长度为4的满足要求的子序列有两种方案，一种是选取第1、2、3、5个({1,3,6,7})，一种是选取第1、2、4、5个({1,3,4,7})。
        因而答案为2
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int[] seq = new int[n];
//        for (int i = 0; i < n; i++) {
//            seq[i] = sc.nextInt();
//        }
//        sc.close();
//
//        getAllSeq(seq,0,new ArrayList<>());
//        System.out.println(allSeq);
//    }
    static List<List<Integer>> allSeq = new ArrayList<>();
    public static void getAllSeq(int[] seq, int cur, List<Integer> s) {
        if (s.size() > 0 && seq[cur] <= s.get(s.size() - 1)) {
            allSeq.add(new ArrayList<>(s));
            return;
        } else s.remove(s.size()-1);

        s.add(seq[cur]);
        for (int i = cur+1; i < seq.length; i++) {
            getAllSeq(seq, i, s);
        }
    }

/*便利蜂2021-04-09*/
    /*
    字符串编码
    时间限制： 3000MS
    内存限制： 589824KB
    题目描述：
    现有字符串编码规则如下两种类型： 数字N字母串 数字N{字母串}
    其中数字N是十进制正整数，大小不超过100 字母串由A到Z的大写字母构成，也有可能为空串
    两种规则均表示将字母串重复N次
    当然也有可能： 只包含数字---则表示这个数字 只包含字母串---则表示这个字母串
    现在希望你实现一个函数把给定字符串解码出来，例如：

    输入：5A 输出：AAAAA
    输入：5{A}B 输出：AAAAAB
    输入：21 输出：21
    输入：{AB}CD 输出：ABCD

    输入描述
    编码前的字符串

    输出描述
    编码后的字符串

    样例输入
    2{A4{C}}
    样例输出
    ACCCCACCCC
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        StringBuilder encoded = new StringBuilder(sc.nextLine().trim());
//        sc.close();
//
//        //配对花括号 通过率 73%
//        int left,right;
//        while (encoded.lastIndexOf("{") != -1) {
//            left = encoded.lastIndexOf("{");
//            right = encoded.indexOf("}",left);
//            String str = encoded.substring(left+1, right).trim();
//
//            if (left-1 >= 0 && encoded.charAt(left-1) >= '0' && encoded.charAt(left-1) <= '9') {
//                int numIdx = left-1;
//                while (numIdx > 0 && encoded.charAt(numIdx) >= '0' && encoded.charAt(numIdx) <= '9') numIdx--;
//
//                if (encoded.charAt(numIdx) < '0' || encoded.charAt(numIdx) > '9') numIdx++;
//                int n = Integer.parseInt(encoded.substring(numIdx,left));
//                StringBuilder tmp = new StringBuilder();
//                for (int i = 0; i < n; i++) {
//                    tmp.append(str);
//                }
//                encoded.replace(numIdx,right+1,tmp.toString());
//            } else {
//                encoded.replace(left,right+1,str);
//            }
//        }
//
//        String decoded = encoded.toString();
//        System.out.println(decoded);
//    }

    /*
    时间限制： 3000MS
    内存限制： 589824KB
    题目描述：
    便利蜂门店每月要将门店的水电费发票邮寄回总部，为了提升管理效率，
    总部财务同学希望统计一下每月预计会在哪些日期段收到指定区域的发票，
    比如北京海淀上地街道有5家门店，8月份预计收到发票的日期如下

    则8月可能收到发票的日期段为 8月3日~8日，8月10日~18日。

    输入描述
    指定月份多家门店收到发票的预计日期区间串，上述案例中的输入为： 3-5,4-8,10-15,17-18,16-17

    输出描述
    返回合并后的日期区间，上述案例输出为：3-8,10-18

    样例输入
    3-5,4-8,18-22,20-24,28-30
    样例输出
    3-8,18-24,28-30

    提示
    1、输入日期区间不必考虑无效情况，如2月30日
    2、所有门店的输入日期区间均不需要考虑跨月情况，如某门店3月28日~4月1日
    3、相邻日期段需要合并

    样例2：
    输入： 2-5,5-7
    输出： 2-7

    样例3：
    输入： 2-5,6-7,8-9
    输出： 2-9
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String dateStr = sc.nextLine().trim();
//        sc.close();
//
//        String[] dates = dateStr.split(",");
//        int[][] dateInts = new int[dates.length][2];
//
//        for (int i = 0; i < dates.length; i++) {
//            int index = dates[i].indexOf("-");
//            dateInts[i][0] = Integer.parseInt(dates[i].substring(0,index));
//            dateInts[i][1] = Integer.parseInt(dates[i].substring(index+1));
//        }

          //对齐时间 通过率91%
//        Arrays.sort(dateInts, (o1, o2) -> {
//            if (o1[0] == o2[0]) return o1[1] - o2[1];
//            return o1[0] - o2[0];
//        });
//
//        StringBuilder result = new StringBuilder();
//        int begin = 0;
//        for (int i = 1; i < dateInts.length; i++) {
//            if (dateInts[i][0] > dateInts[i-1][1]+1) {
//                result.append(dateInts[begin][0]).append("-").append(dateInts[i-1][1]).append(",");
//                begin = i;
//            }
//        }
//
//        result.append(dateInts[begin][0]).append("-").append(dateInts[dateInts.length-1][1]);
//        System.out.println(result);
//    }

    /*
    探测网络质量
    时间限制： 3000MS
    内存限制： 589824KB
    题目描述：
    便利蜂需要对每个门店的网络质量进行监控，以便发现影响业务的网络问题类问题。
    技术人员设计了监控策略：在一台门店设备上以每秒一次的频率使用ping命令探测从门店到www.bianlifeng.com的延时，把探测的结果输出到一个数组中nums。

    为了配置网络异常的报警，需要根据采样区间k找出每个区间内的最大值，采样区间每次向右移动一位，
    需要从左到右从数据nums中依次找到每个采样区间的最大值，放入一个数组中，最终返回网络探测采样区间最大值的数组集合。

    例题：
    输入: nums = [100,300,50,30,500,103,605,720], 和 k = 3 输出:[300,300,500,500,605,720]
    解释:  采样区间的位置最大值
    [100,300,50],30,500,103,605,720
    300 100,[300,50,30],500,103,605,720
    300 100,300,[50,30,500],103,605,720
    500 100,300,50,[30,500,103],605,720
    500 100,300,50,30,[500,103,605],720
    605 100,300,50,30,500,[103,605,720]720
    要求：时间复杂度为 O(n)

    输入描述
    第一行是nums
    第二行是k

    说明：
    nums：网络探测ping耗时结果数组，例如[100,300,50,30,500,103,605,720]
    k：采样区间大小

    输出描述
    返回网络探测采样区间最大值的数组集合

    提示
    你可以假设 k 总是有效的，在输入数组不为空的情况下，1 ≤ k ≤ 输入数组的大小。
    样例2：
    输入：
    600,100,700,30,30,30,60,120
    3
    输出：[700,700,700,30,60,120]
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String numStr = sc.nextLine();
//        int k = sc.nextInt();
//        sc.close();
//
//        String[] numArr = numStr.split(",");
//        int[] nums = new int[numArr.length];
//        for (int n = 0; n < numArr.length; n++) {
//            nums[n] = Integer.parseInt(numArr[n]);
//        }
//
//        //计算出需要采样多少次,创建出存储集合
//        int[] ret = new int[nums.length-k+1];
//        LinkedList<Integer> list = new LinkedList<>();
//        for(int i = 0; i < nums.length; i++) {
//            // 保证单调递增性
//            while(!list.isEmpty() && nums[list.getLast()] <= nums[i]) {
//                list.removeLast();
//            }
//            // 添加当前数的数组下标
//            list.addLast(i);
//            // 先清理掉上次区间遗留的最大值数组下标
//            if (list.getFirst() <= i-k) {
//                list.removeFirst();
//            }
//            // 再保存当前区间中最大值
//            if(i-k+1 >= 0){
//                ret[i-k+1] = nums[list.getFirst()];
//            }
//        }
//
//        System.out.println(Arrays.toString(ret).replace(" ",""));
//    }

/*滴滴2021-04-10*/
    /*
    任务调度
    时间限制： 3000MS
    内存限制： 589824KB
    题目描述：
    优秀的操作系统离不开优秀的任务调度算法。
    现在，有一台计算机即将执行n个任务，每个任务都有一个准备阶段和执行阶段。
    只有在准备阶段完成后，执行阶段才可以开始。
    同一时间，计算机只能执行一个任务的执行阶段，同时可以执行任意多个任务的准备阶段。
    请你设计一个算法，合理分配任务执行顺序，并输出完成所有任务的最少时间。

    输入描述
    第一行一个整数n表示任务的数量（1<=n<=5*10^4）
    接下来n行每行两个整数a，b表示第i个任务的准备时长和执行时长。（1<=a,b<=10^9）
    输出描述
    仅一行一个整数，表示执行所有任务的最少时间。

    样例输入
    2
    5 1
    2 4
    样例输出
    7
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int[][] tasks = new int[n][2];
//
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < 2; j++) {
//                tasks[i][j] = sc.nextInt();
//            }
//        }
//        sc.close();
//
//
//    }

    /*
    施展魔法
    时间限制： 3000MS
    内存限制： 589824KB
    题目描述：
    小A的家门口有一排树，每棵树都有一个正整数的高度。由于树的高度不同，来到小A家的朋友都觉得很难看。
    为了将这些树高度变得好看，小A决定对其中某些树施展魔法，具体来说，每施展一次魔法，可以把一棵树的高度变成任意正整数（可以变高也可以变低）。
    小A认为，这排树如果能构成等差为x的等差数列就好看了。但是小A不想施展太多次魔法，他想知道最少施展魔法的次数。
    形式上来说，小A家门口一共有n棵树，第i棵树高度为ai。
    小A最后的目标为对于任意2≤i≤n，ai-ai-1=x

    输入描述
    输入第一行包含两个正整数，n和x，含义如题面所示。
    输入第二行包含n个正整数，第i个数的含义为第i棵树的高度ai
    范围：n≤105,1≤ai≤105，x≤1000

    输出描述
    输出包含一个正整数，即小A最少施展魔法的次数

    样例输入
    5 2
    1 3 1 3 5
    样例输出
    3
    提示
    对3,4,5号树施法，最后变为1,3,5,7,9
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int x = sc.nextInt();
//        int[] trees = new int[n];
//
//        for (int i = 0; i < n; i++) {
//            trees[i] = sc.nextInt();
//        }
//        sc.close();
//
//
//    }

/*2021-4-12去哪儿*/
    /*
    数组元素
    时间限制： 3000MS
    内存限制： 589824KB
    题目描述：
    给定一个数组，求数组中重复次数第K多的那个元素。
    时间复杂度：O(nlogn)
    空间复杂度：O(n)

    输入描述
        第一行为一个整数N，表示数组的大小；
        接下来N行，每行一个整数，表示数组的每一个元素；
        接下来一行为整数K。
    输出描述
        数组中重复次数第K多的那个元素。

    样例输入
        2
        2
        2
        1
    样例输出
        2
     */
    static int topK(int[] array, int k) {
        TreeMap<Integer,Integer> map = new TreeMap<>();
        for (int j : array) {
            map.put(j, map.getOrDefault(j, 0) + 1);
        }

        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(map::get));
        for (int key : map.keySet()) {
            if (pq.size() < k) pq.add(key);
            else if (map.get(key) > map.get(pq.peek())) {
                pq.remove();
                pq.add(key);
            }
        }

        while (pq.size() > 1) pq.remove();
        return pq.remove();
    }

    /*
    数字加密
    时间限制： 3000MS
    内存限制： 589824KB
    题目描述：
    在某个对数字进行加密的系统，对于x位的数字，会对中间的[start, end)位改为大小写敏感的字母后存储，start最小为0，从左边开始数。
    比如11位的电话号码，会对第[3, 7)进行加密，12312340000加密为123AbCd0000，
    结果一次操作操作失误把号码的所有字母都转换成了小写，比如123AbCd0000变成了123abcd0000。
    给出x, start, end和一个操作失误后的号码，求所有可能的操作失误前的号码
    （以字符序排序后输出，123ABCD0000应为第一个，123abcd0000为最后一个，最后一个输出之后不要加换行符)。



    输入描述
    x start end 操作失误后的号码（以空格分隔）

    输出描述
    每行一个可能的操作失误前的号码

    样例输入
    11 3 7 123abcd0000
    样例输出
    123ABCD0000
    123ABCd0000
    123ABcD0000
    123ABcd0000
    123AbCD0000
    123AbCd0000
    123AbcD0000
    123Abcd0000
    123aBCD0000
    123aBCd0000
    123aBcD0000
    123aBcd0000
    123abCD0000
    123abCd0000
    123abcD0000
    123abcd0000

    提示
    也可以考虑不用length，start，end信息，直接对号码进行操作
     */
    static List<String> alls = new ArrayList<>();
    static String before;
    static String after;
    static List<String> getAll(int length, int start, int end, String input) {
        before = input.substring(0,start);
        after = input.substring(end);

        String encode = input.substring(start,end);

        helperDecode(encode,0,new StringBuilder());

        return alls;
    }

    private static void helperDecode(String encode, int i,StringBuilder builder) {

        if (builder.length() == encode.length()) {

            alls.add(before+builder.toString()+after);
            return;
        }
        char c = encode.charAt(i);
        for (int j = 0; j < 2; j++) {
            if (j == 0) {
                builder.append(Character.toUpperCase(c));
                helperDecode(encode,i+1,builder);
                builder.deleteCharAt(builder.length()-1);
            }
            else {
                builder.append(Character.toLowerCase(c));
                helperDecode(encode,i+1,builder);
                builder.deleteCharAt(builder.length()-1);
            }
        }
    }

    /*
    酒店报价
    时间限制： 3000MS
    内存限制： 589824KB
    题目描述：
    去哪网酒店后端报价数据存储格结构如以下格式所示：
    2021-05-01~2021-05-02:188 表示2021-05-01入住，2021-05-02离店，入住一天的价格是￥188。

    现有多组这样的日期价格段，请将其合并，合并的规则如下：
    1)价格相同，日期段相邻或者重叠的需要合并
    2)相同日期的价格以排在后面的数据为准
    根据指定输入，将所有的日期价格段合并后，按照价格升序排列，价格相同，日期更早的排在前。
    输出格式，每条数据换行输出。
    举例1：
    当输入是：2021-05-01~2021-06-01:388,2021-05-20~2021-06-30:388
    输出结果是：
    2021-05-01~2021-06-30:388
    举例2：
    当输入是：2021-08-01~2021-12-31:388,2021-10-01~2021-10-07:588
    输出结果是：
    2021-08-01~2021-10-01:388
    2021-10-07~2021-12-31:388
    2021-10-01~2021-10-07:588
     */
//    public static void main(String[] args) throws ParseException {
//        Scanner sc = new Scanner(System.in);
//        String timeStrs = sc.nextLine();
//        sc.close();
//
//        String[] timeArr = timeStrs.split(",");
//        Date[][] times = new Date[timeArr.length][2];
//        //存储这段时间内的房间价格
//        Map<Date[],Integer> map = new HashMap<>();
//
//        for (int i = 0; i < timeArr.length; i++) {
//            int jj = timeArr[i].indexOf("~");
//            int dd = timeArr[i].indexOf(":");
//            times[i][0] = SimpleDateFormat.getDateInstance().parse(timeArr[i].substring(0,jj));
//            times[i][1] = SimpleDateFormat.getDateInstance().parse(timeArr[i].substring(jj+1,dd));
//            map.put(times[i],Integer.parseInt(timeArr[i].substring(dd+1)));
//        }
//
//        //对齐数组
//        Arrays.sort(times, (o1, o2) -> {
//            if (o1[0].getTime() == o2[0].getTime()) return (int) (o1[1].getTime() - o2[1].getTime());
//            return (int) (o1[0].getTime() - o2[0].getTime());
//        });
//
//        //先按照时间进行分割,再按照金额进行分割
//
//    }

/*2021-4-17神策数据*/
    /*
    字符串编码,使用字母+数字方式编码       abb->a1b2   aaadccc->a3d1c3
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String decode = sc.nextLine();
//        sc.close();
//        StringBuilder encode = new StringBuilder();
//
//        int idx = 0;
//        for (int i = 0; i < decode.length(); i++) {
//            char c = decode.charAt(i);
//            char d = decode.charAt(idx);
//            if (c != d) {
//                encode.append(d).append(i-idx);
//                idx = i;
//            }
//        }
//        encode.append(decode.charAt(idx)).append(decode.length()-idx);
//
//        System.out.println(encode.toString());
//    }

    /*
    括号配对: 给定一个字符串,打印里面配对的括号的个数和下标 给定的测试用例中不存在无法配对情况
    输入用例: (1)23(4()5)6
    输出用例: 3  共有三个匹配,接下来每两行代表下标,以左括号的下标升序排序
            0
            2
            5
            10
            7
            8
     */
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String strs = sc.nextLine();
//        sc.close();
//
//        Stack<Integer> stack = new Stack<>();
//        PriorityQueue<int[]> results = new PriorityQueue<>(Comparator.comparingInt(o -> o[0]));
//        for (int i = 0; i < strs.length(); i++) {
//            char c = strs.charAt(i);
//            if (c == '(') {
//                stack.push(i);
//            }
//            else if (c == ')') {
//                int left = stack.pop();
//                results.offer(new int[]{left,i});
//            }
//        }
//        System.out.println(results.size());
//        while (!results.isEmpty()) {
//            int[] ints = results.poll();
//            System.out.println(ints[0]);
//            System.out.println(ints[1]);
//        }
//    }

    /*
    字符串全排列: 给定一个假设不重复的字符串 生成一个List 输出所有字符串的全排列
    输入用例: abc
    输出用例: ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
     */
//    static List<String> permutations = new ArrayList<>();
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String unique = sc.nextLine();
//        sc.close();
//
//        helperPermute(unique,new boolean[unique.length()],new StringBuilder());
//        StringBuilder output = new StringBuilder();
//        output.append("[");
//        for (int i = 0; i < permutations.size(); i++) {
//            output.append('\'');
//            output.append(permutations.get(i));
//            output.append('\'');
//            if (i != permutations.size()-1) {
//                output.append(',').append(' ');
//            }
//        }
//        output.append("]");
//
//        System.out.println(output.toString());
//    }
//
//    private static void helperPermute(String unique,boolean[] visited,StringBuilder perm) {
//        if (perm.length() == unique.length()) {
//            permutations.add(perm.toString());
//            return;
//        }
//        for (int i = 0; i < unique.length(); i++) {
//            if (visited[i]) continue;
//            visited[i] = true;
//            perm.append(unique.charAt(i));
//            helperPermute(unique, visited, perm);
//            perm.deleteCharAt(perm.length()-1);
//            visited[i] = false;
//        }
//    }

/*2021-4-25 58同城*/
    /*
    给定一个无序且无重复数字的数组,求出其构成的二叉搜索树的树高
    默认定义数组的第一个数字为树的根节点
     */
    public int tdepth (int[] arr) {
        if (arr == null || arr.length == 0) return 0;

        bst root = new bst(arr[0]);

        //构建二叉搜索树
        for (int i = 1; i < arr.length; i++) {
            buildbst(root,arr[i]);
        }

        //求树深度
        return bstDepth(root);
    }

    private void buildbst(bst root,int val) {
        //去左子树
        if (root.val > val) {
            if (root.left == null) root.left = new bst(val);
            else buildbst(root.left,val);
        }
        //去右子树
        else {
            if (root.right == null) root.right = new bst(val);
            else buildbst(root.right,val);
        }
    }

    private int bstDepth(bst root) {
        if (root == null) return 0;
        return Math.max(bstDepth(root.left),bstDepth(root.right)) + 1;
    }

    static class bst {
        int val;
        bst left;
        bst right;
        bst(int v) {
            this.val = v;
        }
    }

    /*
    最长回文子串
    https://writings.sh/post/algorithm-longest-palindromic-substring
     */
    public String longestPalindrome (String string) {
        if (string == null || string.equals("") || string.length() == 1) return string;

        StringBuilder builder = new StringBuilder(string);

        String longest = "";
//        for (int i = 0; i < builder.length(); i++) {
//            for (int j = builder.length()-1; j > i; j--) {
//                if (isPalindrome(builder,i,j) && j-i > longest.length()) {
//                    longest = builder.substring(i,j+1);
//                }
//            }
//        }
        for (int i = 0; i < builder.length(); i++) {
            // 当回文串是奇数时，由一个中心点向两边扩散
            String s1 = centerExpandFindPalindrome(builder, i, i);
            // 当回文串是偶数时，由中间的两个中心点向两边扩散
            String s2 = centerExpandFindPalindrome(builder, i, i + 1);

            // 三元运算符：判断为真时取冒号前面的值，为假时取冒号后面的值
            longest = longest.length() > s1.length() ? longest : s1;
            longest = longest.length() > s2.length() ? longest : s2;
        }

        return longest;
    }
    //判断是否回文子串
    private boolean isPalindrome(StringBuilder sub,int left,int right) {
        while (left < right) {
            if (sub.charAt(left++) != sub.charAt(right--)) return false;
        }
        return true;
    }
    //中心扩展法
    private String centerExpandFindPalindrome (StringBuilder builder, int left, int right) {
        // 在区间 [0, s.length() - 1] 中寻找回文串，防止下标越界
        while (left >=0 && right < builder.length()) {
            // 是回文串时，继续向两边扩散
            if (builder.charAt(left) == builder.charAt(right)) {
                left--;
                right++;
            } else {
                break;
            }
        }

        // 循环结束时的条件是 s.charAt(left) != s.charAt(right), 所以正确的区间为 [left + 1, right)
        return builder.substring(left + 1, right);
    }

    /*
     合并两个升序数组
     */
    public int[] mergePrice (int[] a, int[] b) {
        if (a == null || a.length == 0) return b;
        if (b == null || b.length == 0) return a;

        int[] result = new int[a.length + b.length];
        int ai = 0;
        int bi = 0;
        for (int i = 0; i < result.length; i++) {
            if (ai == a.length) {
                System.arraycopy(b,bi,result,i,result.length-i);
                break;
            }
            if (bi == b.length) {
                System.arraycopy(a,ai,result,i,result.length-i);
                break;
            }
            if (a[ai] < b[bi]) {
                result[i] = a[ai++];
            }
            else {
                result[i] = b[bi++];
            }
        }

        return result;
    }

}






