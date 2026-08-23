package com.zhifaios.eyes.aispect;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

/** 中文注释：远程模型文件操作只依赖 API 24 可用的 java.io，避免 java.nio.file 的系统版本差异。 */
final class AispectFileIO {
    private AispectFileIO() {
    }

    static byte[] readBytes(File file) throws IOException {
        if (file == null || !file.isFile()) {
            throw new IOException("file_missing");
        }
        try (FileInputStream input = new FileInputStream(file);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    static void writeBytes(File file, byte[] content) throws IOException {
        if (file == null || content == null) {
            throw new IOException("file_write_invalid");
        }
        try (FileOutputStream output = new FileOutputStream(file)) {
            output.write(content);
            output.flush();
        }
    }

    static void copy(File source, File target) throws IOException {
        writeBytes(target, readBytes(source));
    }

    static void moveReplace(File source, File target) throws IOException {
        if (source == null || target == null || !source.exists()) {
            throw new IOException("file_move_invalid");
        }
        if (target.exists() && !target.delete()) {
            throw new IOException("file_move_target_delete_failed");
        }
        if (!source.renameTo(target)) {
            throw new IOException("file_move_failed");
        }
    }
}
