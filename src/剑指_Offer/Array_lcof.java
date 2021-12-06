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
        return true;
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

        }
        return numbers[l];
    }

}
