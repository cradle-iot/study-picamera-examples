import boto3

def insert(items):
    # データベース接続の初期化
    session = boto3.session.Session(
                                    region_name = os.environ['REGION']
                                    aws_access_key_id = os.environ['A_KEY']
                                    aws_secret_access_key = os.environ['S_KEY']
                                    )
    dynamodb = session.resource('dynamodb')


    # テーブルと接続
    table_name = os.environ['TABLE_NAME']
    table = dynamodb.Table(table_name)

    for item in items:
        # 追加する
        response = table.put_item(
            TableName=table_name,
            Item=item
        )
        if response['ResponseMetadata']['HTTPStatusCode'] is not 200:
            # 失敗処理
            print(response)
        else:
            # 成功処理
            print('Successed :', item['device'])
    return
