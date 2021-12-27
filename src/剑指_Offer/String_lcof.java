package 剑指_Offer;

/**
 * 字符串类题目
 */
public class String_lcof {

    // 6. 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
    public String replaceSpace(String s) {
        if (s == null) {
            return s;
        }
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == ' ') {
                result.append("%20");
            } else {
                result.append(c);
            }
        }
        return result.toString();
    }

    //7. 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。
    public String reverseLeftWords(String s, int n) {
        return s.substring(n) + s.substring(0,n);
    }

    //13. 第一个只出现一次的字符
    // 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。
    public char firstUniqChar(String s) {
        if (s == null || s.length() == 0) {
            return ' ';
        }
        if (s.length() == 1) {
            return s.charAt(0);
        }

        //遍历计数 时间O(N) 空间O(1)
        int[] chars = new int[26];
        for (int i = 0; i < s.length(); i++) {
            chars[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (chars[c - 'a'] == 1) {
                return c;
            }
        }

        return ' ';
    }

    // 剑指 Offer 58 - I. 翻转单词顺序
    // 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。
    // 为简单起见，标点符号和普通字母一样处理。
    // 例如输入字符串"I am a student. "，则输出"student. a am I"。
    //来源：力扣（LeetCode）
    //链接：https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof
    public String reverseWords(String s) {
        if (s == null) {
            return null;
        }
        s = s.trim();
        if (s.length() == 0) {
            return "";
        }
        StringBuilder result = new StringBuilder();

        //解法一 双指针解法 时间O(N^2) 空间O(1)
        /*int l = s.length()-1, r = s.length();
        while (l > 0) {
            if (s.charAt(l) == ' ') {
                result.append(s, l+1, r).append(' ');
                while (l > 0 && s.charAt(l) == ' ') {
                    r = l;
                    l--;
                }
            } else {
                l--;
            }
        }
        result.append(s, l, r);*/

        //解法二 空间换时间 时间O(N) 空间O(N)
        String[] ss =s.trim().split(" ");
        for(int i = ss.length-1; i>=0; i--) {
            if("".equals(ss[i])) {
                //注意这里是空而不是空格
                continue;
            }
            result.append(" ").append(ss[i]);
        }

        return result.toString().trim();
    }

    // 剑指 Offer 67. 把字符串转换成整数
    // 写一个函数 StrToInt，实现把字符串转换成整数这个功能。
    // 不能使用 atoi 或者其他类似的库函数。
    // 首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
    //当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；
    // 假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
    // 该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
    // 注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
    // 在任何情况下，若函数不能进行有效的转换时，请返回0。
    // 说明： 假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为[−2^31, 2^(31−1)]。
    //       如果数值超过这个范围，请返回 INT_MAX (2^(31−1)) 或INT_MIN (−2^31)。
    // 来源：力扣（LeetCode）
    // 链接：https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof
    public int strToInt(String str) {
        if (str == null || "".equals(str)) {
            return 0;
        }
        int j = 0;
        while (j < str.length() && str.charAt(j) == ' ') j++;
        if (j == str.length()) return 0;
        char firChar = str.charAt(j);
        if (!Character.isDigit(firChar) && firChar != '-' && firChar != '+' ) {
            return 0;
        }
        boolean flag = firChar == '-';
        long result = 0;
        for (int i = firChar == '+' || firChar == '-' ? j+1:j; i < str.length(); i++) {
            char c = str.charAt(i);
            if (Character.isDigit(c)) {
                result = result*10 + c-'0';
                if (!flag && result > Integer.MAX_VALUE) {
                    return Integer.MAX_VALUE;
                } else if (flag && -result < Integer.MIN_VALUE) {
                    return Integer.MIN_VALUE;
                }
            } else break;
        }
        return (int) (flag ? -result : result);
    }

    // 剑指 Offer 20. 表示数值的字符串
    // 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
    // 来源：力扣（LeetCode）
    // 链接：https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/
    public boolean isNumber(String s) {
        if (s == null || s.length() == 0) return false;
        s = s.trim();
        boolean numFlag = false;
        boolean dotFlag = false;
        boolean eFlag = false;
        for (int i = 0; i < s.length(); i++) {
            //判定为数字 则标记numFlag
            if (s.charAt(i) >= '0' && s.charAt(i) <= '9') numFlag = true;
            //判定为.  需要没出现过.并且没出现过e
            else if (s.charAt(i) == '.' && !dotFlag && !eFlag) dotFlag = true;
            //判定为e  需要没出现过e,并且出过数字了
            else if ((s.charAt(i) == 'e' || s.charAt(i) == 'E') && !eFlag && numFlag) {
                eFlag = true;
                numFlag = false;  //避免123e形式,出现e之后就标志为false
            }
            //判定为+-符号 只能出现在第一位或者紧接e后面
            else if ((s.charAt(i) == '+' || s.charAt(i) == '-') && (i == 0 || s.charAt(i - 1) == 'e' || s.charAt(i - 1) == 'E')) {}
            //其他情况,都是非法的
            else return false;
        }
        return numFlag;
    }

}
