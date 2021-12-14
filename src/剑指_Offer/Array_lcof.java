package 剑指_Offer;

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

}
