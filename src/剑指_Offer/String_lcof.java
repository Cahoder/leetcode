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

}
