import os

root_dir = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data_few_shot"

def rename_images_in_folder(folder_path):
    images_dir = os.path.join(folder_path, "images")
    if not os.path.exists(images_dir):
        print(f"[BỎ QUA] {folder_path} - không có thư mục 'images'")
        return

    # Lấy danh sách file ảnh
    valid_exts = {".png", ".jpg", ".jpeg"}
    images = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    images.sort()  # sắp xếp tên gốc để giữ thứ tự hợp lý

    if not images:
        print(f"[BỎ QUA] {folder_path} - không tìm thấy ảnh")
        return

    obj_name = os.path.basename(folder_path)

    print(f"[ĐANG XỬ LÝ] {obj_name} ({len(images)} ảnh)")

    for idx, filename in enumerate(images, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{obj_name}_{idx:04d}{ext}"
        src = os.path.join(images_dir, filename)
        dst = os.path.join(images_dir, new_name)
        os.rename(src, dst)

    print(f"→ Đổi tên xong {len(images)} ảnh trong {obj_name}")

def main():
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            rename_images_in_folder(folder_path)

if __name__ == "__main__":
    main()
