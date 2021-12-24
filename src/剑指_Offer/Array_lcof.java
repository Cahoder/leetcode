package 剑指_Offer;

import java.util.*;

/**
 * 数组类题目
 */
public class Array_lcof {

    //11. 二维数组中的查找
    // 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
    // 请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
    // 来源：力扣（LeetCode）
    // 链接：https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }

        //解法一 迭代方式 矩阵右上角开始二分查找 时间O(n+m) 空间O(1)
        int x = matrix[0].length - 1;
        int y = 0;
        while ( x >= 0 && y < matrix.length) {
            int i = matrix[y][x];
            if (i == target) {
                return true;
            } else if (i > target) {
                x--;
            } else {
                y++;
            }
        }

        //解法二 递归方式 矩阵右上角开始二叉搜索树解法

        return false;
    }

    //12. 旋转数组的最小数字
    // 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
    // 给你一个可能存在重复元素值的数组numbers，它原来是一个升序排列的数组，并按上述情形进行了一次旋转。
    // 请返回旋转数组的最小元素。
    // 例如，数组[3,4,5,1,2] 为 [1,2,3,4,5] 的一次旋转，该数组的最小值为1。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof
    public int minArray(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return -1;
        }

        //解法一 从后往前遍历 时间O(N) 空间O(1)
        /*for (int i = numbers.length - 1; i > 0; i--) {
            int number = numbers[i];
            if (number < numbers[i-1]) {
                return number;
            }
        }
        return numbers[0];*/

        //解法二 二分查找 时间O(NLogN) 空间O(1)
        int l = 0, r = numbers.length - 1;
        while (l < r) {
            int m = l + ((r - l) >> 1);
            //如果中间比右边大，肯定还在升序中
            if (numbers[m] > numbers[r]) {
                l = m + 1;
            }
            else if (numbers[m] < numbers[r]) {
                r = m;
            }
            // 去重
            else {
                r--;
            }
        }
        return numbers[l];
    }

    // 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
    // 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
    // 使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。
    public int[] exchange(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return nums;
        }

        //解法一 遍历交换 时间O(N^2) 空间O(1)
        /*for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            //如果当前是偶数
            if ((num & 1) == 0) {
                int j = i+1;
                for (; j < nums.length; j++) {
                    int num1 = nums[j];
                    //遇到奇数前移
                    if ((num1 & 1) == 1) {
                        nums[i] = num1;
                        nums[j] = num;
                        j = i;
                        break;
                    }
                }
                i = j;
            }
        }*/

        //解法二 左右双指针向中间聚集（快排思想） 时间O(N) 空间O(1)
        int l = 0, r = nums.length-1, tmp;
        while (l < r) {
            //左指针遇到偶数停下来
            while (l < r && (nums[l] & 1) == 1) {
                l++;
            }
            //右指针遇到奇数停下来
            while (l < r && (nums[r] & 1) == 0) {
                r--;
            }
            if (l < r) {
                tmp = nums[l];
                nums[l] = nums[r];
                nums[r] = tmp;
                l++;
                r--;
            }
        }

        return nums;
    }

    // 剑指 Offer 57. 和为s的两个数字
    // 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。
    // 如果有多对数字的和等于s，则输出任意一对即可。
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length < 2) {
            return new int[0];
        }
        //解法一 哈希表 时间O(N) 空间O(N)
        // !set.contains(target - nums[i]) ? set.add(nums[i]) : return new int[]{nums[i], target - nums[i]}

        //解法二 双撞指针 时间O(N) 空间O(1)
        //先利用递增排序的规律缩小范围
        int r = 0;
        for (; r < nums.length; r++) {
            if (nums[r] >= target) break;
        }
        //防止右指针越界
        r--;
        //如果不足两位数则凑不出答案
        if (r < 1) {
            return new int[0];
        }
        //利用左右指针求和规律: (l + r > target) ? r-- : l++;
        int l = 0;
        while (l < r) {
            int sum = nums[l] + nums[r];
            if (sum == target) {
                return new int[] {nums[l], nums[r]};
            } else if (sum > target) {
                r--;
            } else {
                l++;
            }
        }

        return new int[0];
    }

    // 给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
    // 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
    // 你可以假设除了整数 0 之外，这个整数不会以零开头。
    // https://leetcode-cn.com/problems/plus-one/
    public int[] plusOne(int[] digits) {
        if(digits == null || digits.length == 0) return digits;

        if (digits[digits.length-1] != 9) {
            digits[digits.length-1]++;
            return digits;
        }

        int i = digits.length-1;
        while (digits[i] == 9) {
            i--;
            if (i == 0) break;
        }

        if (i == 0 && digits[i] == 9) {
            digits = new int[digits.length+1];
            digits[0] = 1;
        } else {
            digits[i]++;
            for (i++; i < digits.length; i++) {
                digits[i] = 0;
            }
        }

        return digits;
    }

    // 剑指 Offer 56 - I. 数组中数字出现的次数
    // 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。
    // 请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
    // 限制：2 <= nums.length <= 10000
    // https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/
    public int[] singleNumbers(int[] nums) {
        if (nums == null || nums.length < 2) {
            return new int[0];
        }

        //解法一 映射计数 时间O(n) 空间O(n)
        /*int[] result = new int[2];
        Map<Integer,Integer> numCountMap = new HashMap<>();
        int idx = 0;
        for (int num : nums) {
            numCountMap.put(num, numCountMap.getOrDefault(num,0)+1);
        }
        for (Integer num : numCountMap.keySet()) {
            if (idx == result.length) {
                break;
            }
            Integer count = numCountMap.get(num);
            if (count == 1) {
                result[idx++] = num;
            }
        }
        return result;*/

        //解法二 异或位分组异或 时间O(n) 空间O(1)
        /*
        相同的数异或为0，不同的异或为1。0和任何数异或等于这个数本身。
        所以，数组里面所有数异或 = 目标两个数异或（由于这两个数不同，所以异或结果必然不为0）
        假设数组异或的二进制结果为10010，那么说明这两个数从右向左数第2位是不同的
        那么可以根据数组里面所有数的第二位为0或者1将数组划分为2个。
        这样做可以将两个目标数分散在不同的数组中，
        这两个数组里面的数各自进行异或，得到的结果就是答案
         */
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        //找数组里数之间异或位的idx
        //idx可通过xor&(-xor)直接获得
        int idx = 0;
        while ((xor & 1) == 0) {
            idx++;
            xor >>= 1;
        }
        //在异或位各自进行异或，得到的就是答案
        int diff1 = 0, diff2 = 0;
        for (int num : nums) {
            if ((num >> idx & 1) == 0) {
                diff1 ^= num;
            }
            else if ((num >> idx & 1) == 1) {
                diff2 ^= num;
            }
        }
        return new int[] {diff1, diff2};
    }

    // 剑指 Offer 56 - II. 数组中数字出现的次数 II
    // 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。
    // 请找出那个只出现一次的数字。
    // 限制：1 <= nums.length <= 10000  &&  1 <= nums[i] < 2^31
    // https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length < 4) {
            return nums[0];
        }
        //解法一 映射计数 时间O(n) 空间O(n)
        // TODO ignored

        //解法二 排序 时间O(nLogn) 空间O(1)
        /*Arrays.sort(nums);
        for(int i =0; i <= nums.length-2; i += 3) {
            if(nums[i] != nums[i+2]) {
                return nums[i];
            }
        }
        return nums[nums.length-1];*/

        //解法三 位运算 时间O(32n) 空间O(1)
        int res = 0, i = 0;
        while (i < 32) {
            int bit = 0;
            for (int num : nums) {
                bit += ((num >> i) & 1);
            }
            res += (bit % 3 << i++);
        }
        return res;
    }

    // 剑指 Offer 39. 数组中出现次数超过一半的数字
    // 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
    // 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
    // 限制：1 <= 数组长度 <= 50000
    // https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/
    public int majorityElement(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length < 3) {
            return nums[0];
        }

        //解法一 映射计数 时间O(n) 空间O(n)
        // TODO ignored

        //解法二 排序 时间O(nLogn) 空间O(1)
        /*Arrays.sort(nums);
        int result = nums[0];
        int maxCount = 1;
        int count = 1;
        for(int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i-1]) {
                count = 1;
            }
            count++;
            if (count > maxCount) {
                maxCount = count;
                result = nums[i];
            }
        }
        return result;*/

        //解法三 排序后取中位数(超过一半的数排序后必定占据中间) 时间O(nLogn) 空间O(1)
        /*Arrays.sort(nums);
        return nums[nums.length >> 1];*/

        //解法四 摩尔投票法(极限一换一,最后活下来的肯定是多的) 时间O(n) 空间O(1)
        int count = 0;
        int result = 0;
        for (int num : nums) {
            if (count == 0) result = num;
            count += (result == num) ? 1:-1;
        }
        return result;
    }

    // 剑指 Offer 66. 构建乘积数组
    // 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，
    // 其中B[i]的值是数组A中除了下标 i 以外的元素的积,
    // 即B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
    //
    // 提示： 所有元素乘积之和不会溢出 32 位整数 && a.length <= 100000
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof
    public int[] constructArr(int[] a) {
        if (a == null || a.length < 1) {
            return new int[0];
        }
        if (a.length == 1) {
            return new int[]{0};
        }
        int[] result = new int[a.length];

        //解法一 暴力遍历 时间O(n^2) 空间O(1)
        /*for (int i = 0; i < a.length; i++) {
            int mult = 1;
            for (int j = 0; j < a.length; j++) {
                if (i != j) {
                    mult *= a[j];
                }
            }
            result[i] = mult;
        }*/

        //解法二 错位累乘 时间O(n) 空间O(1)
        for (int i = 0, mult = 1; i < a.length; i++) {
            result[i] = mult;
            mult *= a[i];
        }
        for (int i = a.length - 1, mult = 1; i >= 0; i--) {
            result[i] *= mult;
            mult *= a[i];
        }

        return result;
    }

}
