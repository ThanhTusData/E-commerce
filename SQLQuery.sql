-- 1. Nhập Thư Viện
-- Không cần trong SQL

-- 2. Đọc Dữ Liệu
-- Trong SQL, dữ liệu được truy xuất từ bảng (giả sử bảng tên là `orders`)
SELECT *
INTO #df_initial
FROM orders;

-- 3. Chuyển Đổi Dữ Liệu Thời Gian
--ALTER TABLE #df_initial
--ADD InvoiceDate DATETIME;

UPDATE #df_initial
SET InvoiceDate = CONVERT(DATETIME, InvoiceDate, 120); -- Căn cứ vào định dạng thời gian thực tế

-- 4. Kiểm Tra Thông Tin Cột
SELECT 
    COLUMN_NAME, 
    DATA_TYPE, 
    (SELECT COUNT(*) FROM #df_initial WHERE COLUMN_NAME IS NULL) AS null_values_nb, 
    (SELECT COUNT(*) FROM #df_initial WHERE COLUMN_NAME IS NULL) * 100.0 / (SELECT COUNT(*) FROM #df_initial) AS null_values_percentage
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'df_initial';  -- Sửa tên bảng tạm nếu cần

-- 5. Xóa Giá Trị Null
DELETE FROM #df_initial WHERE CustomerID IS NULL;

-- 6. Kiểm Tra và Xóa Bản Ghi Trùng Lặp
DELETE FROM #df_initial
WHERE InvoiceDate NOT IN (
    SELECT MIN(InvoiceDate) 
    FROM #df_initial 
    GROUP BY InvoiceNo, CustomerID, StockCode, Quantity, UnitPrice, Description
);

-- 7. Phân Tích Theo Quốc Gia
SELECT 
    Country, 
    COUNT(*) AS OrderCount
FROM #df_initial
GROUP BY Country;

-- 8. Thống Kê Sản Phẩm và Khách Hàng
SELECT 
    COUNT(DISTINCT StockCode) AS Products,
    COUNT(DISTINCT InvoiceNo) AS Transactions,
    COUNT(DISTINCT CustomerID) AS Customers
FROM #df_initial;

-- 9. Phân Tích Chi Tiết Đơn Hàng
SELECT 
    CustomerID, 
    InvoiceNo, 
    COUNT(*) AS Number_of_products
FROM #df_initial
GROUP BY CustomerID, InvoiceNo;

-- 10. Tính Số Lượng Đơn Hàng Bị Hủy
SELECT 
    SUM(CASE WHEN InvoiceNo LIKE '%C%' THEN 1 ELSE 0 END) AS CanceledOrders,
    COUNT(*) AS TotalOrders
FROM (
    SELECT 
        CustomerID, 
        InvoiceNo
    FROM #df_initial
    GROUP BY CustomerID, InvoiceNo
) AS subquery;

-- 11. Kiểm Tra Các Đơn Hàng Với Số Lượng Âm
SELECT *
FROM #df_initial
WHERE Quantity < 0;

-- 12. Làm Sạch Dữ Liệu
-- Tạo bảng tạm #df_cleaned từ #df_initial
SELECT *
INTO #df_cleaned
FROM #df_initial;

-- Thêm cột QuantityCanceled
ALTER TABLE #df_cleaned
ADD QuantityCanceled INT DEFAULT 0;

-- Tạo một bảng tạm để lưu trữ các bản ghi cần cập nhật
CREATE TABLE #UpdateList (
    CustomerID VARCHAR(255),
    StockCode VARCHAR(255),
    QuantityCanceled INT
);

-- Khai báo biến để sử dụng trong vòng lặp
DECLARE @CustomerID VARCHAR(255);
DECLARE @StockCode VARCHAR(255);
DECLARE @Quantity INT;

-- Kiểm tra xem con trỏ có tồn tại hay không trước khi tạo
IF OBJECT_ID('tempdb..#entry_cursor') IS NOT NULL
BEGIN
    DEALLOCATE entry_cursor; -- Giải phóng nếu con trỏ đã tồn tại
END

-- Con trỏ để duyệt qua các bản ghi có Quantity < 0
DECLARE entry_cursor CURSOR FOR 
SELECT CustomerID, StockCode, Quantity
FROM #df_cleaned
WHERE Quantity < 0;

OPEN entry_cursor;

FETCH NEXT FROM entry_cursor INTO @CustomerID, @StockCode, @Quantity;

WHILE @@FETCH_STATUS = 0
BEGIN
    -- Xác định số lượng bản ghi trong df_cleaned có Quantity > 0
    DECLARE @df_test_count INT;
    SELECT @df_test_count = COUNT(*)
    FROM #df_cleaned
    WHERE CustomerID = @CustomerID AND StockCode = @StockCode AND Quantity > 0;

    IF @df_test_count = 1
    BEGIN
        INSERT INTO #UpdateList (CustomerID, StockCode, QuantityCanceled)
        VALUES (@CustomerID, @StockCode, -@Quantity);  -- Đảm bảo rằng số lượng tương ứng với các cột
    END
    ELSE IF @df_test_count > 1
    BEGIN
        -- Cập nhật QuantityCanceled cho bản ghi phù hợp
        UPDATE TOP(1) #df_cleaned
        SET QuantityCanceled = -@Quantity
        WHERE CustomerID = @CustomerID AND StockCode = @StockCode AND Quantity >= -@Quantity
        AND InvoiceDate IN (
            SELECT TOP 1 InvoiceDate 
            FROM #df_cleaned 
            WHERE CustomerID = @CustomerID AND StockCode = @StockCode AND Quantity > 0 
            ORDER BY InvoiceDate DESC
        );
    END

    FETCH NEXT FROM entry_cursor INTO @CustomerID, @StockCode, @Quantity;
END

CLOSE entry_cursor;
DEALLOCATE entry_cursor;

-- Cập nhật các bản ghi đã lưu trong bảng tạm
UPDATE #df_cleaned
SET QuantityCanceled = (SELECT QuantityCanceled FROM #UpdateList WHERE #UpdateList.CustomerID = #df_cleaned.CustomerID AND #UpdateList.StockCode = #df_cleaned.StockCode)
WHERE EXISTS (SELECT 1 FROM #UpdateList WHERE #UpdateList.CustomerID = #df_cleaned.CustomerID AND #UpdateList.StockCode = #df_cleaned.StockCode);

-- Xóa bảng tạm sau khi sử dụng
DROP TABLE #UpdateList;

-- 13. Tính Giá Trị Tổng
ALTER TABLE #df_cleaned
ADD TotalPrice DECIMAL(10, 2);

UPDATE #df_cleaned
SET TotalPrice = UnitPrice * (Quantity - ISNULL(QuantityCanceled, 0));  -- Sử dụng ISNULL để tránh lỗi

-- 14. Đếm Số Lượng Mua Hàng
SELECT 
    PriceRange, 
    COUNT(*) AS OrderCount
FROM (
    SELECT *,
        CASE 
            WHEN TotalPrice BETWEEN 0 AND 50 THEN '0-50'
            WHEN TotalPrice BETWEEN 51 AND 100 THEN '51-100'
            WHEN TotalPrice BETWEEN 101 AND 200 THEN '101-200'
            WHEN TotalPrice BETWEEN 201 AND 300 THEN '201-300'
            WHEN TotalPrice BETWEEN 301 AND 400 THEN '301-400'
            WHEN TotalPrice BETWEEN 401 AND 500 THEN '401-500'
            WHEN TotalPrice BETWEEN 501 AND 600 THEN '501-600'
            WHEN TotalPrice BETWEEN 601 AND 700 THEN '601-700'
            WHEN TotalPrice BETWEEN 701 AND 800 THEN '701-800'
            WHEN TotalPrice BETWEEN 801 AND 900 THEN '801-900'
            WHEN TotalPrice BETWEEN 901 AND 1000 THEN '901-1000'
        END AS PriceRange
    FROM #df_cleaned
) AS temp
GROUP BY PriceRange;

-- 15. Lưu Dữ Liệu
-- Không cần trong SQL; bạn có thể lưu dữ liệu từ bảng tạm vào bảng chính nếu cần.
SELECT * INTO cleaned_data 
FROM #df_cleaned;
