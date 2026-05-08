import os
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
import warnings
import shutil

warnings.filterwarnings('ignore')


class UrbanBuiltupProcessor:
    def __init__(self, original_data_root, intermediate_root, final_output_root, admin_shp_path):
        self.original_data_root = original_data_root  # 原数据文件夹
        self.intermediate_root = intermediate_root  # 中间结果文件夹
        self.final_output_root = final_output_root  # 最终结果文件夹
        self.admin_shp_path = admin_shp_path
        self.admin_gdf = None
        self.area_field = 'Area'

        # 确保输出目录存在
        os.makedirs(self.intermediate_root, exist_ok=True)
        os.makedirs(self.final_output_root, exist_ok=True)

    def load_admin_data(self):
        """加载行政区数据"""
        print("正在加载行政区数据...")
        self.admin_gdf = gpd.read_file(self.admin_shp_path)
        print(f"行政区数据加载完成，共 {len(self.admin_gdf)} 个行政区")

    def get_city_folders(self):
        """获取所有城市文件夹"""
        folders = []
        for item in os.listdir(self.original_data_root):
            item_path = os.path.join(self.original_data_root, item)
            if os.path.isdir(item_path):
                folders.append(item)
        return sorted(folders)

    def get_city_shp_files(self, city_folder, data_root):
        """获取城市文件夹中的所有建成区shp文件"""
        city_path = os.path.join(data_root, city_folder)
        shp_files = []
        for file in os.listdir(city_path):
            if file.endswith('_11_built_metro_edge.shp') and not file.endswith('_admin_union.shp'):
                shp_files.append(file)
        return sorted(shp_files)

    def extract_year_from_filename(self, filename):
        """从文件名中提取年份"""
        parts = filename.split('_')
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return int(part)
        return None

    def get_utm_zone(self, geometry):
        """根据几何体质心计算UTM带"""
        centroid = geometry.centroid
        lon, lat = centroid.x, centroid.y

        # 计算UTM zone
        utm_zone = int((lon + 180) / 6) + 1

        # 确定南北半球
        if lat >= 0:
            epsg_code = 32600 + utm_zone
        else:
            epsg_code = 32700 + utm_zone

        return f"EPSG:{epsg_code}"

    def find_intersecting_admin(self, city_gdf, city_name):
        """查找与建成区有交集的行政区"""
        print(f"正在检测 {city_name} 建成区与行政区的交集...")

        # 合并所有年份的建成区用于交集检测
        union_geometry = city_gdf.unary_union

        # 创建临时几何体用于交集计算
        temp_union = gpd.GeoDataFrame(geometry=[union_geometry], crs=city_gdf.crs)

        # 关键修改：将建成区数据转换到WGS84进行相交检测
        temp_union_wgs84 = temp_union.to_crs(epsg=4326)  # 转换为WGS84

        intersecting_admin = []

        for idx, admin_row in self.admin_gdf.iterrows():
            admin_geom = admin_row.geometry

            # 检查是否有交集（现在都在WGS84坐标系下）
            if temp_union_wgs84.geometry[0].intersects(admin_geom):
                # 为了计算准确的交集面积，需要在投影坐标系下计算
                # 将行政区几何体转换到建成区的UTM坐标系
                admin_utm = gpd.GeoSeries([admin_geom], crs=4326).to_crs(city_gdf.crs).iloc[0]
                intersection = temp_union.geometry[0].intersection(admin_utm)

                if not intersection.is_empty:
                    area = intersection.area
                    # 关键修改：保存完整的行政区信息，包括dt_adcode用于唯一标识
                    intersecting_admin.append({
                        'dt_name': admin_row['dt_name'],
                        'dt_adcode': admin_row['dt_adcode'],
                        'pr_name': admin_row['pr_name'],
                        'ct_name': admin_row['ct_name'],
                        'area_km2': area / 1e6  # 转换为平方公里
                    })

        # 按交集面积降序排列
        intersecting_admin.sort(key=lambda x: x['area_km2'], reverse=True)
        return intersecting_admin

    def create_admin_union(self, selected_indices, intersecting_admin, city_folder, city_name):
        """创建选择的行政区的合并文件"""
        # 关键修改：使用dt_adcode而不是dt_name来唯一标识行政区
        selected_dt_adcodes = [intersecting_admin[i]['dt_adcode'] for i in selected_indices]
        selected_dt_names = [intersecting_admin[i]['dt_name'] for i in selected_indices]

        # 筛选选择的行政区 - 使用dt_adcode确保唯一性
        selected_admin = self.admin_gdf[self.admin_gdf['dt_adcode'].isin(selected_dt_adcodes)]

        if len(selected_admin) == 0:
            print("警告：未找到选择的行政区，请检查名称匹配")
            return None

        # 合并行政区
        union_geometry = selected_admin.unary_union
        union_gdf = gpd.GeoDataFrame(geometry=[union_geometry], crs=self.admin_gdf.crs)

        # 保存合并后的行政区到中间结果文件夹
        output_city_path = os.path.join(self.intermediate_root, city_folder)
        os.makedirs(output_city_path, exist_ok=True)
        output_path = os.path.join(output_city_path, f"{city_name}_admin_union.shp")
        union_gdf.to_file(output_path, encoding='utf-8')

        print(f"已创建行政区合并文件: {output_path}")
        return selected_dt_names

    def process_city_year_step1(self, shp_file, city_folder, admin_union_gdf, city_name):
        """处理单个年份的建成区文件 - 步骤1：行政区裁剪和碎斑处理"""
        input_file_path = os.path.join(self.original_data_root, city_folder, shp_file)
        output_city_path = os.path.join(self.intermediate_root, city_folder)
        os.makedirs(output_city_path, exist_ok=True)
        output_file_path = os.path.join(output_city_path, shp_file)

        try:
            # 读取建成区数据
            builtup_gdf = gpd.read_file(input_file_path)

            if len(builtup_gdf) == 0:
                print(f"警告: {shp_file} 为空文件，跳过处理")
                # 复制空文件到输出目录
                builtup_gdf.to_file(output_file_path, encoding='utf-8')
                return False

            print(f"原始数据CRS: {builtup_gdf.crs}")

            # 关键修改：将行政区合并数据转换到建成区的CRS进行裁剪
            admin_union_proj = admin_union_gdf.to_crs(builtup_gdf.crs)

            # 裁剪建成区
            clipped_gdf = gpd.clip(builtup_gdf, admin_union_proj)

            if len(clipped_gdf) == 0:
                print(f"警告: {shp_file} 裁剪后无数据")
                # 创建空文件到输出目录
                empty_gdf = gpd.GeoDataFrame(geometry=[], crs=builtup_gdf.crs)
                empty_gdf.to_file(output_file_path, encoding='utf-8')
                return True

            # 获取适合面积计算的CRS（已经是UTM投影，直接使用）
            area_crs = builtup_gdf.crs
            print(f"面积计算使用的CRS: {area_crs}")

            # 计算面积并过滤小斑块
            clipped_gdf['area_temp'] = clipped_gdf.geometry.area
            filtered_gdf = clipped_gdf[clipped_gdf['area_temp'] >= 10000]

            if len(filtered_gdf) == 0:
                print(f"警告: {shp_file} 过滤后无数据")
                # 创建空文件到输出目录
                empty_gdf = gpd.GeoDataFrame(geometry=[], crs=builtup_gdf.crs)
                empty_gdf.to_file(output_file_path, encoding='utf-8')
                return True

            # 合并所有多边形
            dissolved_geometry = filtered_gdf.unary_union

            # 创建新的GeoDataFrame
            result_gdf = gpd.GeoDataFrame(geometry=[dissolved_geometry], crs=filtered_gdf.crs)

            # 计算总面积并添加到字段
            total_area = filtered_gdf['area_temp'].sum()
            result_gdf[self.area_field] = total_area

            # 保存到中间结果文件夹
            result_gdf.to_file(output_file_path, encoding='utf-8')

            print(f"步骤1完成: {shp_file}, 总面积: {total_area:.2f} m²")
            return True

        except Exception as e:
            print(f"处理文件 {shp_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def apply_constraint(self, target_gdf, constraint_gdf):
        """修正约束逻辑：目标年份的建成区应该是约束年份建成区的子集"""
        print(f"  应用约束: 目标年份要素数={len(target_gdf)}, 约束年份要素数={len(constraint_gdf)}")

        # 如果约束年份没有建成区，那么目标年份也应该没有建成区
        if len(constraint_gdf) == 0:
            print("  约束年份无建成区，目标年份全部被裁剪")
            return gpd.GeoDataFrame(geometry=[], crs=target_gdf.crs)

        # 如果目标年份没有建成区，直接返回空
        if len(target_gdf) == 0:
            print("  目标年份无建成区，直接返回空")
            return target_gdf

        # 关键修正：目标年份的建成区应该是约束年份建成区的子集
        # 使用约束年份的建成区来裁剪目标年份的建成区
        # 只保留目标年份中在约束年份建成区范围内的部分
        try:
            # 使用空间交集操作，只保留目标年份中与约束年份建成区重叠的部分
            result = gpd.overlay(target_gdf, constraint_gdf, how='intersection')

            print(f"  约束后要素数: {len(result)}")
            return result

        except Exception as e:
            print(f"  空间分析出错: {str(e)}")
            # 如果空间分析失败，尝试使用更简单的方法
            try:
                # 使用裁剪操作
                result = gpd.clip(target_gdf, constraint_gdf)
                print(f"  使用裁剪方法约束后要素数: {len(result)}")
                return result
            except Exception as e2:
                print(f"  备用空间分析也出错: {str(e2)}")
                # 如果所有方法都失败，返回空结果
                return gpd.GeoDataFrame(geometry=[], crs=target_gdf.crs)

    def recalculate_area(self, gdf):
        """重新计算面积并更新Area字段"""
        # 检查是否存在Area字段，如果不存在则创建
        if 'Area' not in gdf.columns:
            gdf['Area'] = 0.0

        # 计算每个多边形的面积（单位：平方米）
        gdf['Area'] = gdf.geometry.area

        return gdf

    def process_city_year_step2(self, shp_file, city_folder, constraint_gdf, city_name, target_years):
        """处理单个年份的建成区文件 - 步骤2：年份约束处理"""
        input_file_path = os.path.join(self.intermediate_root, city_folder, shp_file)
        output_city_path = os.path.join(self.final_output_root, city_folder)
        os.makedirs(output_city_path, exist_ok=True)

        # 提取年份
        year = self.extract_year_from_filename(shp_file)
        if year is None:
            print(f"无法从文件名 {shp_file} 中提取年份，跳过")
            return False

        # 输出文件名
        output_filename = f"{city_folder}_{year}_constraint.shp"
        output_path = os.path.join(output_city_path, output_filename)

        try:
            # 读取数据
            year_gdf = gpd.read_file(input_file_path)
            original_area = year_gdf['Area'].sum() if len(year_gdf) > 0 and 'Area' in year_gdf.columns else 0
            print(f"处理年份 {year}，原始要素数量: {len(year_gdf)}, 原始面积: {original_area:.2f} m²")

            # 如果是目标年份，应用约束
            if year in target_years:
                print(f"年份 {year}: 应用约束")

                result_gdf = self.apply_constraint(year_gdf, constraint_gdf)

                # 重新计算面积
                if len(result_gdf) > 0:
                    result_gdf = self.recalculate_area(result_gdf)
                    new_area = result_gdf['Area'].sum()
                    result_gdf.to_file(output_path, encoding='utf-8')
                    print(f"  约束后保留 {len(result_gdf)} 个要素，约束后面积: {new_area:.2f} m²")
                    print(
                        f"  面积变化: {new_area - original_area:.2f} m² ({((new_area / original_area) - 1) * 100:.2f}%)")
                else:
                    # 创建空的GeoDataFrame
                    empty_gdf = gpd.GeoDataFrame(geometry=[], crs=year_gdf.crs)
                    empty_gdf['Area'] = []
                    empty_gdf.to_file(output_path, encoding='utf-8')
                    print(f"  约束后无要素保留，面积为0")

            else:
                # 非目标年份，直接复制并重命名
                print(f"年份 {year}: 直接复制")
                year_gdf.to_file(output_path, encoding='utf-8')
                print(f"  复制 {len(year_gdf)} 个要素，面积: {original_area:.2f} m²")

            print(f"  已保存: {output_filename}")
            return True

        except Exception as e:
            print(f"处理年份 {year} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def get_admin_selection(self, intersecting_admin, city_folder):
        """获取行政区选择"""
        print(f"\n请为城市 {city_folder} 选择要保留的行政区")
        print("注意：您可以慢慢研究，程序会等待您的输入")
        print("输入 'skip' 可以跳过此城市")
        print("输入 'quit' 可以退出程序")

        while True:
            try:
                choice_input = input("\n请输入要保留的行政区编号（例如: 0 或 0,1,2）: ").strip()

                # 添加特殊命令处理
                if choice_input.lower() == 'skip':
                    print(f"跳过城市 {city_folder}")
                    return None
                elif choice_input.lower() == 'quit':
                    print("用户选择退出程序")
                    exit(0)
                elif not choice_input:
                    print("未选择任何行政区")
                    return None

                selected_indices = []
                for part in choice_input.split(','):
                    part = part.strip()
                    if part:
                        idx = int(part)
                        if 0 <= idx < len(intersecting_admin):
                            selected_indices.append(idx)
                        else:
                            print(f"编号 {idx} 超出范围，请重新输入")
                            break
                else:
                    if selected_indices:
                        selected_names = [intersecting_admin[i]['dt_name'] for i in selected_indices]
                        print(f"已选择: {selected_names}")
                        confirm = input("确认选择？(y/n): ").strip().lower()
                        if confirm == 'y':
                            return selected_indices
                        else:
                            print("重新选择...")
                    else:
                        print("未选择任何行政区")
            except ValueError:
                print("输入格式错误，请使用数字编号，如: 0 或 0,1,2")
            except KeyboardInterrupt:
                print("\n用户中断")
                return None

    def get_constraint_rules(self, city_folder, available_years):
        """获取约束规则"""
        print(f"\n处理城市: {city_folder}")
        print(f"可用的年份: {available_years}")
        print("\n请选择约束规则类型:")
        print("1. 手动指定约束规则")
        print("2. 逐期约束（每期建成区按照下一期的来约束）")
        print("3. 不应用约束（仅复制文件）")
        print("输入 'quit' 退出程序")

        while True:
            try:
                choice = input("\n请选择约束规则类型 (1/2/3): ").strip()

                if choice.lower() == 'quit':
                    print("用户选择退出程序")
                    exit(0)

                if choice == '1':
                    return self.get_manual_constraint_rules(city_folder, available_years)
                elif choice == '2':
                    return self.get_sequential_constraint_rules(city_folder, available_years)
                elif choice == '3':
                    return {'skip': True}
                else:
                    print("输入错误，请选择 1, 2 或 3")

            except KeyboardInterrupt:
                print("\n用户中断")
                return {'skip': True}

    def get_manual_constraint_rules(self, city_folder, available_years):
        """获取手动指定的约束规则"""
        print(f"\n手动指定约束规则")
        print("请输入约束规则，格式为: 约束年份 目标年份1 目标年份2 ...")
        print("例如: 2025 2010 2015 2020 表示2010、2015、2020年按照2025年进行约束")
        print("约束逻辑: 目标年份的建成区应该是约束年份建成区的子集")
        print("输入 'back' 返回上级菜单")
        print("输入 'quit' 退出程序")

        while True:
            try:
                user_input = input("\n请输入约束规则: ").strip()

                if user_input.lower() == 'back':
                    return self.get_constraint_rules(city_folder, available_years)
                elif user_input.lower() == 'quit':
                    print("用户选择退出程序")
                    exit(0)
                elif not user_input:
                    print("输入为空，请重新输入")
                    continue

                # 解析输入
                parts = user_input.split()
                if len(parts) < 2:
                    print("输入格式错误，至少需要两个年份")
                    continue

                # 解析约束年份
                constraint_year = int(parts[0])
                if constraint_year < 100:
                    constraint_year += 2000

                # 解析目标年份
                target_years = []
                for part in parts[1:]:
                    year = int(part)
                    if year < 100:
                        year += 2000
                    target_years.append(year)

                # 验证年份是否存在
                if constraint_year not in available_years:
                    print(f"约束年份 {constraint_year} 不存在，请重新输入")
                    continue

                for year in target_years:
                    if year not in available_years:
                        print(f"目标年份 {year} 不存在，请重新输入")
                        continue

                return {
                    'constraint_type': 'manual',
                    'constraint_year': constraint_year,
                    'target_years': target_years,
                    'skip': False
                }

            except ValueError:
                print("输入格式错误，请使用数字表示年份")
            except KeyboardInterrupt:
                print("\n用户中断")
                return {'skip': True}

    def get_sequential_constraint_rules(self, city_folder, available_years):
        """获取逐期约束规则"""
        print(f"\n逐期约束规则")
        print("每期建成区按照下一期的来约束:")

        # 对年份进行排序
        sorted_years = sorted(available_years)

        # 生成逐期约束规则
        constraint_pairs = []
        for i in range(len(sorted_years) - 1):
            current_year = sorted_years[i]
            next_year = sorted_years[i + 1]
            constraint_pairs.append((current_year, next_year))

        print("自动生成的约束规则:")
        for current_year, next_year in constraint_pairs:
            print(f"  {current_year}年 -> 用{next_year}年约束")

        print(f"注意: {sorted_years[-1]}年没有后续年份，不应用约束")
        print("约束逻辑: 前期建成区应该是后期建成区的子集")

        print("\n是否应用此约束规则?")
        confirm = input("确认应用逐期约束? (y/n): ").strip().lower()

        if confirm == 'y':
            return {
                'constraint_type': 'sequential',
                'constraint_pairs': constraint_pairs,
                'skip': False
            }
        else:
            print("取消逐期约束，返回上级菜单")
            return self.get_constraint_rules(city_folder, available_years)

    def copy_files_without_constraint(self, city_folder, available_years, year_to_file):
        """跳过约束处理时，复制中间结果文件到最终输出文件夹"""
        print(f"跳过城市 {city_folder} 的约束处理，仅复制文件")

        # 确保最终输出城市文件夹存在
        final_city_path = os.path.join(self.final_output_root, city_folder)
        os.makedirs(final_city_path, exist_ok=True)

        # 中间结果城市文件夹路径
        intermediate_city_path = os.path.join(self.intermediate_root, city_folder)

        # 复制所有文件到最终文件夹并重命名
        for year in available_years:
            input_file = year_to_file[year]
            input_path = os.path.join(intermediate_city_path, input_file)

            # 输出文件名
            output_filename = f"{city_folder}_{year}_constraint.shp"
            output_path = os.path.join(final_city_path, output_filename)

            try:
                # 检查输入文件是否存在
                if not os.path.exists(input_path):
                    print(f"警告: 中间结果文件不存在: {input_path}")
                    # 创建一个空的shapefile
                    empty_gdf = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
                    empty_gdf['Area'] = []
                    empty_gdf.to_file(output_path, encoding='utf-8')
                    print(f"  已创建空文件: {output_filename}")
                    continue

                # 读取并保存数据
                year_gdf = gpd.read_file(input_path)
                year_gdf.to_file(output_path, encoding='utf-8')
                print(f"  已复制: {output_filename}")

            except Exception as e:
                print(f"复制年份 {year} 时出错: {str(e)}")
                # 即使出错也继续处理其他年份

        print(f"城市 {city_folder} 文件复制完成")
        return True

    def process_sequential_constraint(self, city_folder, constraint_rules, year_to_file):
        """处理逐期约束"""
        constraint_pairs = constraint_rules['constraint_pairs']

        print(f"应用逐期约束规则:")
        for current_year, next_year in constraint_pairs:
            print(f"  {current_year}年 -> 用{next_year}年约束")

        # 确保最终输出城市文件夹存在
        final_city_path = os.path.join(self.final_output_root, city_folder)
        os.makedirs(final_city_path, exist_ok=True)

        # 中间结果城市文件夹路径
        intermediate_city_path = os.path.join(self.intermediate_root, city_folder)

        success_count = 0
        total_years = len(year_to_file)

        # 处理每个约束对
        for current_year, constraint_year in constraint_pairs:
            current_file = year_to_file[current_year]
            constraint_file = year_to_file[constraint_year]

            current_path = os.path.join(intermediate_city_path, current_file)
            constraint_path = os.path.join(intermediate_city_path, constraint_file)

            output_filename = f"{city_folder}_{current_year}_constraint.shp"
            output_path = os.path.join(final_city_path, output_filename)

            try:
                # 读取当前年份和约束年份的数据
                current_gdf = gpd.read_file(current_path)
                constraint_gdf = gpd.read_file(constraint_path)

                original_area = current_gdf['Area'].sum() if len(
                    current_gdf) > 0 and 'Area' in current_gdf.columns else 0
                print(f"\n处理 {current_year}年 (用{constraint_year}年约束):")
                print(f"  原始要素数量: {len(current_gdf)}, 原始面积: {original_area:.2f} m²")

                # 应用约束
                result_gdf = self.apply_constraint(current_gdf, constraint_gdf)

                # 重新计算面积
                if len(result_gdf) > 0:
                    result_gdf = self.recalculate_area(result_gdf)
                    new_area = result_gdf['Area'].sum()
                    result_gdf.to_file(output_path, encoding='utf-8')
                    print(f"  约束后保留 {len(result_gdf)} 个要素，约束后面积: {new_area:.2f} m²")
                    print(
                        f"  面积变化: {new_area - original_area:.2f} m² ({((new_area / original_area) - 1) * 100:.2f}%)")
                else:
                    # 创建空的GeoDataFrame
                    empty_gdf = gpd.GeoDataFrame(geometry=[], crs=current_gdf.crs)
                    empty_gdf['Area'] = []
                    empty_gdf.to_file(output_path, encoding='utf-8')
                    print(f"  约束后无要素保留，面积为0")

                print(f"  已保存: {output_filename}")
                success_count += 1

            except Exception as e:
                print(f"处理年份 {current_year} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        # 处理最后一个年份（没有约束）
        all_years = sorted(year_to_file.keys())
        last_year = all_years[-1]
        last_file = year_to_file[last_year]
        last_path = os.path.join(intermediate_city_path, last_file)
        output_filename = f"{city_folder}_{last_year}_constraint.shp"
        output_path = os.path.join(final_city_path, output_filename)

        try:
            last_gdf = gpd.read_file(last_path)
            area = last_gdf['Area'].sum() if len(last_gdf) > 0 and 'Area' in last_gdf.columns else 0
            last_gdf.to_file(output_path, encoding='utf-8')
            print(f"\n处理 {last_year}年 (无约束，直接复制):")
            print(f"  复制 {len(last_gdf)} 个要素，面积: {area:.2f} m²")
            print(f"  已保存: {output_filename}")
            success_count += 1
        except Exception as e:
            print(f"处理年份 {last_year} 时出错: {str(e)}")

        return success_count == total_years

    def process_city(self, city_folder):
        """处理单个城市 - 合并步骤1和步骤2"""
        original_city_path = os.path.join(self.original_data_root, city_folder)

        # 检查是否已完成处理 - 使用代码1的断点检测方式
        done_file = os.path.join(original_city_path, '_FINISHED_CLIP_AND_MERGE.done')
        choice_file = os.path.join(original_city_path, '_CHOICE.json')

        # 如果两个文件都存在，则跳过该城市
        if os.path.exists(done_file) and os.path.exists(choice_file):
            print(f"城市 {city_folder} 已处理完成，跳过")
            return True

        print(f"\n{'=' * 50}")
        print(f"开始处理城市: {city_folder}")
        print(f"{'=' * 50}")

        # 获取所有建成区文件
        shp_files = self.get_city_shp_files(city_folder, self.original_data_root)
        if not shp_files:
            print(f"在 {city_folder} 中未找到建成区文件，跳过")
            return True

        print(f"找到 {len(shp_files)} 个建成区文件: {shp_files}")

        # 提取可用年份
        available_years = []
        year_to_file = {}
        for shp_file in shp_files:
            year = self.extract_year_from_filename(shp_file)
            if year is not None:
                available_years.append(year)
                year_to_file[year] = shp_file

        available_years.sort()
        print(f"可用年份: {available_years}")

        # 步骤1：行政区裁剪和碎斑处理
        print(f"\n--- 步骤1: 行政区裁剪和碎斑处理 ---")

        # 读取所有建成区文件
        city_builtup_list = []
        for shp_file in shp_files:
            file_path = os.path.join(original_city_path, shp_file)
            try:
                gdf = gpd.read_file(file_path)
                if len(gdf) > 0:
                    city_builtup_list.append(gdf)
                    print(f"  {shp_file}: CRS = {gdf.crs}")
            except Exception as e:
                print(f"读取文件 {shp_file} 失败: {str(e)}")

        if not city_builtup_list:
            print(f"无法读取 {city_folder} 的任何建成区文件，跳过")
            return True

        # 合并所有年份的建成区
        city_combined = gpd.GeoDataFrame(pd.concat(city_builtup_list, ignore_index=True))

        # 查找有交集的行政区
        intersecting_admin = self.find_intersecting_admin(city_combined, city_folder)

        if not intersecting_admin:
            print(f"{city_folder} 的建成区与任何行政区无交集，跳过")
            return True

        # 显示交集行政区列表
        print(f"\n{city_folder} 的建成区与以下行政区有交集:")
        for i, item in enumerate(intersecting_admin):
            # 如果有同名行政区，显示更多信息以区分
            same_name_count = sum(1 for x in intersecting_admin if x['dt_name'] == item['dt_name'])
            if same_name_count > 1:
                print(
                    f"{i}: {item['dt_name']} (编码: {item['dt_adcode']}, 城市: {item['ct_name']}, 省份: {item['pr_name']}, 交集面积: {item['area_km2']:.2f} km²)")
            else:
                print(f"{i}: {item['dt_name']} (交集面积: {item['area_km2']:.2f} km²)")

        # 获取行政区选择
        selected_indices = self.get_admin_selection(intersecting_admin, city_folder)
        if selected_indices is None:
            print("未选择任何行政区，跳过该城市")
            return True

        # 创建行政区合并文件
        selected_dt_names = self.create_admin_union(selected_indices, intersecting_admin, city_folder, city_folder)
        if not selected_dt_names:
            return False

        # 读取行政区合并文件用于裁剪
        intermediate_city_path = os.path.join(self.intermediate_root, city_folder)
        admin_union_path = os.path.join(intermediate_city_path, f"{city_folder}_admin_union.shp")
        admin_union_gdf = gpd.read_file(admin_union_path)

        # 处理每个年份的建成区文件 - 步骤1
        success_count_step1 = 0
        for shp_file in shp_files:
            if self.process_city_year_step1(shp_file, city_folder, admin_union_gdf, city_folder):
                success_count_step1 += 1

        print(f"步骤1完成: {success_count_step1}/{len(shp_files)} 个文件成功处理")

        # 步骤2：年份约束处理
        print(f"\n--- 步骤2: 年份约束处理 ---")

        # 获取约束规则
        constraint_rules = self.get_constraint_rules(city_folder, available_years)

        # 检查是否是跳过约束处理
        if constraint_rules.get('skip', False):
            # 使用新的复制方法
            self.copy_files_without_constraint(city_folder, available_years, year_to_file)

            # 保存选择到JSON文件
            selected_dt_adcodes = [intersecting_admin[i]['dt_adcode'] for i in selected_indices]
            with open(choice_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'selected_dt_adcode': selected_dt_adcodes,
                    'selected_dt_name': selected_dt_names,
                    'constraint_rules': {'skip': True}
                }, f, ensure_ascii=False, indent=2)

            # 创建完成标记文件
            with open(done_file, 'w') as f:
                f.write('completed')

            return True

        # 处理逐期约束
        if constraint_rules.get('constraint_type') == 'sequential':
            success = self.process_sequential_constraint(city_folder, constraint_rules, year_to_file)

            if success:
                # 保存选择到JSON文件
                selected_dt_adcodes = [intersecting_admin[i]['dt_adcode'] for i in selected_indices]
                with open(choice_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'selected_dt_adcode': selected_dt_adcodes,
                        'selected_dt_name': selected_dt_names,
                        'constraint_rules': constraint_rules
                    }, f, ensure_ascii=False, indent=2)

                # 创建完成标记文件
                with open(done_file, 'w') as f:
                    f.write('completed')

                print(f"逐期约束处理完成")
                return True
            else:
                print(f"逐期约束处理失败")
                return False

        # 处理手动约束规则
        constraint_year = constraint_rules['constraint_year']
        target_years = constraint_rules['target_years']

        print(f"约束规则: {target_years} 按照 {constraint_year} 进行约束")

        # 读取约束年份的数据
        constraint_file = year_to_file.get(constraint_year)
        if not constraint_file:
            print(f"约束年份 {constraint_year} 的文件不存在，跳过")
            return False

        constraint_path = os.path.join(intermediate_city_path, constraint_file)
        try:
            constraint_gdf = gpd.read_file(constraint_path)
            print(f"已加载约束年份 {constraint_year} 的数据，包含 {len(constraint_gdf)} 个要素")
        except Exception as e:
            print(f"读取约束年份文件失败: {str(e)}")
            return False

        # 处理每个年份的建成区文件 - 步骤2
        success_count_step2 = 0
        for shp_file in shp_files:
            if self.process_city_year_step2(shp_file, city_folder, constraint_gdf, city_folder, target_years):
                success_count_step2 += 1

        print(f"步骤2完成: {success_count_step2}/{len(shp_files)} 个文件成功处理")

        # 保存选择到JSON文件
        selected_dt_adcodes = [intersecting_admin[i]['dt_adcode'] for i in selected_indices]
        with open(choice_file, 'w', encoding='utf-8') as f:
            json.dump({
                'selected_dt_adcode': selected_dt_adcodes,
                'selected_dt_name': selected_dt_names,
                'constraint_rules': constraint_rules
            }, f, ensure_ascii=False, indent=2)

        # 创建完成标记文件
        with open(done_file, 'w') as f:
            f.write('completed')

        print(f"城市 {city_folder} 完全处理完成")
        return True

    def process_all_cities(self):
        """处理所有城市"""
        city_folders = self.get_city_folders()
        print(f"找到 {len(city_folders)} 个城市文件夹: {city_folders}")

        processed_count = 0
        skipped_count = 0
        error_count = 0

        for i, city_folder in enumerate(city_folders):
            print(f"\n\n=== 进度: {i + 1}/{len(city_folders)} - {city_folder} ===")
            try:
                result = self.process_city(city_folder)
                if result:
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"处理城市 {city_folder} 时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue

        print(f"\n处理完成！")
        print(f"成功处理: {processed_count} 个城市")
        print(f"跳过: {skipped_count} 个城市")
        print(f"错误: {error_count} 个城市")


def main():
    # 配置路径
    original_data_root = r"D:\Thepenger\建成区\检查\刘畅结果\LC第三次重做"  # 原数据
    intermediate_root = r"D:\Thepenger\建成区\检查\刘畅结果\LC第三次重做\建成区裁剪合并结果"  # 中间结果
    final_output_root = r"D:\Thepenger\建成区\检查\刘畅结果\LC第三次重做\建成区约束后结果"  # 最终结果
    admin_shp_path = r"D:\Thepenger\建成区\行政矢量\district.shp"

    # 检查输入路径是否存在
    if not os.path.exists(original_data_root):
        print(f"错误: 原数据目录不存在: {original_data_root}")
        return

    if not os.path.exists(admin_shp_path):
        print(f"错误: 行政区文件不存在: {admin_shp_path}")
        return

    # 创建处理器实例
    processor = UrbanBuiltupProcessor(original_data_root, intermediate_root, final_output_root, admin_shp_path)

    # 加载行政区数据
    processor.load_admin_data()

    # 处理所有城市
    processor.process_all_cities()


if __name__ == "__main__":
    main()