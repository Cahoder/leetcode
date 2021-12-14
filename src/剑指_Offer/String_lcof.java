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

}
